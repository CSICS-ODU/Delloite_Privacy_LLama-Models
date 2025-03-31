# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
import uuid 
import math
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn

from ..api import ModelArgs

# **NOTE**: This code is not runnable without installing `torch` and `fairscale`
# dependencies. These dependencies are not part of the default dependencies
# (requirements.txt) of the `llama-models` package.


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, layer_id:int, args: ModelArgs):
        super().__init__()
        self.layer_id=layer_id
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        layer_id:int
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

         ################## using Sin function
        start = 2
        num_amplify = 100  # Number of scores to amplify
        amplification_value = 1000  # Amplification factor

        if seqlen > 1 and layer_id > 25:
            # Split into start, amplification part, and rest
            start_part = scores[:, :, :, :start]  # Shape: (batch, n_heads, seq_len, start)
            middle_part = scores[:, :, :, start:num_amplify]  # Shape: (batch, n_heads, seq_len, num_amplify-start)
            rest_part = scores[:, :, :, num_amplify:]  # Shape: (batch, n_heads, seq_len, total_len-num_amplify)

            # Generate a sine wave that oscillates between 0 and 1
            seq_length = middle_part.size(-1)
            sine_wave = torch.sin(torch.linspace(0, math.pi, seq_length))  # Generate sine values from 0 to pi
            sine_wave = (sine_wave + 1) / 2  # Scale and shift to [0, 1]

            # Reshape the sine wave to match the dimensions of middle_part
            sine_wave = sine_wave.view(1, 1, 1, -1).expand_as(middle_part)

            # Apply the sine wave to the middle part
            middle_part = (sine_wave * amplification_value)

            # Concatenate the amplified parts back
            scores = torch.cat((start_part, middle_part, rest_part), dim=-1)

        # #  ##################
        # start = 28
        # num_amplify = 55  # Number of scores to amplify
        # amplification_value = 20  # Amplification factor


        # if seqlen > 1 and layer_id >25 :
          
        #     # Split into start, amplification part, and rest
        #     start_part = scores[:, :, :, :start]  # Shape: (batch, n_heads, seq_len, start)
        #     middle_part= scores[:, :, :, start:num_amplify]  # Shape: (batch, n_heads, seq_len, num_amplify-start)
        #     rest_part = scores[:, :, :, num_amplify:] 
        #     amplification_tensor = torch.full_like(middle_part, amplification_value) # Shape: (batch, n_heads, seq_len, total_len-num_amplify)

        #     middle_part=amplification_tensor
 

        #     # Concatenate the amplified parts back
        #     scores = torch.cat((start_part, middle_part, rest_part), dim=-1)
       
        # print(scores.shape)


        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        #  ################## using Sin function
        # start = 28
        # num_amplify = 55  # Number of scores to amplify
        # amplification_value = 1  # Amplification factor

        # if seqlen > 1 and layer_id > 25:
        #     # Split into start, amplification part, and rest
        #     start_part = scores[:, :, :, :start]  # Shape: (batch, n_heads, seq_len, start)
        #     middle_part = scores[:, :, :, start:num_amplify]  # Shape: (batch, n_heads, seq_len, num_amplify-start)
        #     rest_part = scores[:, :, :, num_amplify:]  # Shape: (batch, n_heads, seq_len, total_len-num_amplify)

        #     # Generate a sine wave that oscillates between 0 and 1
        #     seq_length = middle_part.size(-1)
        #     sine_wave = torch.sin(torch.linspace(0, math.pi, seq_length))  # Generate sine values from 0 to pi
        #     sine_wave = (sine_wave + 1) / 2  # Scale and shift to [0, 1]

        #     # Reshape the sine wave to match the dimensions of middle_part
        #     sine_wave = sine_wave.view(1, 1, 1, -1).expand_as(middle_part)

        #     # Apply the sine wave to the middle part
        #     middle_part = (sine_wave * amplification_value)*0.5

        #     # Concatenate the amplified parts back
        #     scores = torch.cat((start_part, middle_part, rest_part), dim=-1)

        #  ##################
        # start = 28
        # num_amplify = 55  # Number of scores to amplify
        # amplification_value = 2.5  # Amplification factor


        # if seqlen > 1 and layer_id >25 :
          
        #     # Split into start, amplification part, and rest
        #     start_part = scores[:, :, :, :start]  # Shape: (batch, n_heads, seq_len, start)
        #     middle_part= scores[:, :, :, start:num_amplify]  # Shape: (batch, n_heads, seq_len, num_amplify-start)
        #     rest_part = scores[:, :, :, num_amplify:] 
        #     amplification_tensor = torch.full_like(middle_part, amplification_value) # Shape: (batch, n_heads, seq_len, total_len-num_amplify)

        #     middle_part=amplification_tensor
 

        #     # Concatenate the amplified parts back
        #     scores = torch.cat((start_part, middle_part, rest_part), dim=-1)

        if seqlen>1 and layer_id == 27:  # Only visualize for the first layer for simplicity
            plt.figure(figsize=(20, 20))
            
            # Extract the middle part of the scores (keys from 2 to 100)
            middle_part = scores[:, :, :, 1:300]  # Shape: (batch, n_heads, seq_len, 98)
            
            # Convert scores to float32 and extract the first head's scores
            scores_float32 = middle_part[0, 0].float().cpu().detach().numpy()
            
            # Focus on queries 0 to 100
            scores_sliced = scores_float32[1:300, :]  # Slice the first 100 queries
            
            # Plot the heatmap for the sliced scores
            sns.heatmap(scores_sliced, cmap='viridis',annot=False)  # Plot the sliced scores
            plt.title(f"Attention Scores (Layer {layer_id}, Head 0) - Queries 0-100, Keys 2-100")
            plt.xlabel("Keys (27-54)")
            plt.ylabel("Queries (27-54)")
            
            # Generate a unique filename
            filename = f"Sin wave attention Amp value 1.png"
            plt.savefig(filename)  # Save the plot to a file
            plt.close()  # Close the plot to free up memory
            print(f"Saved plot to {filename}")# Close the plot to free up memory

        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(layer_id,args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        layer_id:int
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask,layer_id)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = VocabParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        print(params.n_layers)
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
            params.use_scaled_rope,
        )

    @torch.inference_mode() 
    def forward(self, tokens: torch.Tensor, start_pos: int, systemPrompttokens: torch.Tensor,mitigateAdversalPrompt:bool):
        _bsz, seqlen = tokens.shape

       
        # Token embeddings for input tokens
        h = self.tok_embeddings(tokens)
        
        # Token embeddings for the system prompt tokens
        systemPromptEmbeddings = self.tok_embeddings(systemPrompttokens)
  
        
        # Move freqs_cis to the same device as h
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

            if mask.device.type == torch.device("mps").type:
                mask = torch.nan_to_num(mask, nan=0.0)

            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)
        
        
        counter=0
        # Loop through layers and conditionally add system prompt embeddings
        for layer_id,layer in enumerate(self.layers):
            # Add system prompt embeddings only if h.shape matches the condition
            
            if seqlen>1 and mitigateAdversalPrompt:
                
                

                
                # Ensure dimensions match for addition
                if systemPromptEmbeddings.shape[1] != h.shape[1]:
                    if systemPromptEmbeddings.shape[1] < h.shape[1]:
                        # Pad system prompt embeddings
                        diff = h.shape[1] - systemPromptEmbeddings.shape[1]
                        systemPromptEmbeddings = torch.nn.functional.pad(
                            systemPromptEmbeddings, (0, 0, 0, diff), value=0
                        )
                    else:
                        # Slice system prompt embeddings
                        systemPromptEmbeddings = systemPromptEmbeddings[:, :h.shape[1], :]
                # if counter>14:
                #   for i in range(30):

                #     h = h + systemPromptEmbeddings
                # counter+=1

            h = layer(h, start_pos, freqs_cis, mask,layer_id)
           
        # Apply normalization and output layer
        h = self.norm(h)
        output = self.output(h).float()
        
        return output
