<p align="center">
  <img src="/Llama_Repo.jpeg" width="400"/>
</p>

# Privacy-preserving techniques in Large Language Models (LLMs)

This repository showcases an approach to mitigate **adversarial prompts** in **Llama models**. The **System Prompt** includes specialized **Defender Prompts** designed to enhance **privacy** and **security** during inference to achieve this. 

The repository includes a file named `dataset.py`, which contains over **100 synthetically generated comments** designed to replicate **Reddit posts and comments**. These comments are generated to mirror **real-world data**, enabling the testing of models in a simulated environment.

However, a key concern arises from the use of **Adversarial Prompts**. **Open-source models**, when exposed to such prompts, have the potential to infer sensitive personal attributes of users, including but not limited to **location**, **race**, and other **private information**. The adversarial nature of these prompts can lead to unintended inferences about the users' personal characteristics, posing **privacy risks**.

This issue underscores the importance of carefully controlling the types of **inputs** that are fed into open-source models, particularly in environments where **user data** might be inferred from seemingly innocuous comments.

You can read more about the **Privacy Inference attacks** with a link to this paper: [Beyond the Horizon](https://files.sri.inf.ethz.ch/website/papers/staab2023beyond.pdf).


# **Key Changes in the Model**

### **Defender Prompt Embedding:**
The **Defender Prompt** is embedded into the model's processing pipeline during input processing in the last half of the model's layers. This ensures that the **System Prompt's** context is reinforced, especially in later stages where the model might otherwise lose its influence.

<p align="center">
  <img src="/Transformer Archetecture.png" width="400"/>
  **Transformer Architecture Discussed in the "Attention is All You Need" Paper**
</p>

The **Llama 8B-Instruct model** is built on a **Transformer architecture** with **32 layers of decoders**. In this enhanced version, we are embedding the **Defender Prompt** into the model's processing pipeline, specifically in the **last 14 layers** (from layer 19 to layer 32). This **key modification** strengthens the model's **contextual understanding** and ensures that the **system’s prompt** remains influential throughout the entire inference process.



# **Privacy-Aligned Results:**
By incorporating the **Defender Prompt** embedding into the latter layers, the model prioritizes **privacy-preserving mechanisms** while maintaining **contextual relevance** and **accuracy**. This modification ensures outputs that are more aligned with **privacy objectives**, significantly improving the model's **robustness** against **adversarial prompts**.

<p align="center">
  <img src="/Image.png" width="400"/>
</p>



# Llama Models

Llama is an accessible, open large language model (LLM) designed for developers, researchers, and businesses to build, experiment, and responsibly scale their generative AI ideas. Part of a foundational system, it serves as a bedrock for innovation in the global community. A few key aspects:
1. **Open access**: Easy accessibility to cutting-edge large language models, fostering collaboration and advancements among developers, researchers, and organizations
2. **Broad ecosystem**: Llama models have been downloaded hundreds of millions of times, there are thousands of community projects built on Llama and platform support is broad from cloud providers to startups - the world is building with Llama!
3. **Trust & safety**: Llama models are part of a comprehensive approach to trust and safety, releasing models and tools that are designed to enable community collaboration and encourage the standardization of the development and usage of trust and safety tools for generative AI

Our mission is to empower individuals and industry through this opportunity while fostering an environment of discovery and ethical AI advancements. The model weights are licensed for researchers and commercial entities, upholding the principles of openness.

## Llama Models

[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-models)](https://pypi.org/project/llama-models/)
[![Discord](https://img.shields.io/discord/1257833999603335178)](https://discord.gg/TZAAYNVtrU)

|  **Model** | **Launch date** | **Model sizes** | **Context Length** | **Tokenizer** | **Acceptable use policy**  |  **License** | **Model Card** |
| :----: | :----: | :----: | :----:|:----:|:----:|:----:|:----:|
| Llama 2 | 7/18/2023 | 7B, 13B, 70B | 4K | Sentencepiece | [Use Policy](models/llama2/USE_POLICY.md) | [License](models/llama2/LICENSE) | [Model Card](models/llama2/MODEL_CARD.md) |
| Llama 3 | 4/18/2024 | 8B, 70B | 8K | TikToken-based | [Use Policy](models/llama3/USE_POLICY.md) | [License](models/llama3/LICENSE) | [Model Card](models/llama3/MODEL_CARD.md) |
| Llama 3.1 | 7/23/2024 | 8B, 70B, 405B | 128K | TikToken-based | [Use Policy](models/llama3_1/USE_POLICY.md) | [License](models/llama3_1/LICENSE) | [Model Card](models/llama3_1/MODEL_CARD.md) |
| Llama 3.2 | 9/25/2024 | 1B, 3B | 128K | TikToken-based | [Use Policy](models/llama3_2/USE_POLICY.md) | [License](models/llama3_2/LICENSE) | [Model Card](models/llama3_2/MODEL_CARD.md) |
| Llama 3.2-Vision | 9/25/2024 | 11B, 90B | 128K | TikToken-based | [Use Policy](models/llama3_2/USE_POLICY.md) | [License](models/llama3_2/LICENSE) | [Model Card](models/llama3_2/MODEL_CARD_VISION.md) |

## Download

To download the model weights and tokenizer:

1. Visit the [Meta Llama website](https://llama.meta.com/llama-downloads/).
2. Read and accept the license.
3. Once your request is approved you will receive a signed URL via email.
4. Install the [Llama CLI](https://github.com/meta-llama/llama-stack): `pip install llama-stack`. (**<-- Start Here if you have received an email already.**)
5. Run `llama model list` to show the latest available models and determine the model ID you wish to download. **NOTE**:
If you want older versions of models, run `llama model list --show-all` to show all the available Llama models.

6. Run: `llama download --source meta --model-id CHOSEN_MODEL_ID`
7. Pass the URL provided when prompted to start the download.

Remember that the links expire after 24 hours and a certain amount of downloads. You can always re-request a link if you start seeing errors such as `403: Forbidden`.

## Running the models

After cloning this repository, you can execute the .ipynb file to run the open-source Llama models. The output displays the accuracy of the Defender Mechanism both before and after mitigating adversarial prompts.




