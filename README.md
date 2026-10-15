# Exploring Mechanistic Interpretability: A Greek Case Study on Gemma-2B

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An empirical investigation into **Mechanistic Interpretability** and **Representation Steering** using Sparse Autoencoders (SAEs) on Google's **Gemma-2-2B**[cite: 1, 2]. 

This project explores how high-dimensional, polysemantic representations in large language models can be decomposed into human-interpretable feature dictionaries[cite: 1, 2]. By probing the residual stream and applying a pre-trained SAE from **GemmaScope**, we locate a distinct Greek language feature in layer 23 and intervene in model activations to steer generation toward Greek concepts without explicit prompting[cite: 1, 6].

---

## 📌 Key Highlights

- **Logit Lens Probing:** Traced residual stream representations layer-by-layer via the unembedding matrix, detecting an abrupt semantic transition toward Greek token generation between Layer 22 and Layer 23[cite: 5, 6].
- **Sparse Feature Extraction:** Decomposed Layer 23 activations using a 16k-width canonical SAE from GemmaScope, identifying **Feature #8057** as strongly aligned with Greek linguistic and conceptual tokens[cite: 1, 6].
- **External Feature Validation:** Cross-referenced active latent dimensions with [Neuronpedia](https://www.neuronpedia.org/) to verify semantic specificity[cite: 1, 6].
- **Intervention & Steering:** Injected steering vectors directly into the residual stream during forward passes, demonstrating fine-grained control over model outputs across varied coefficients[cite: 1, 6].

---

## 🔬 Methodology

### 1. Identifying the Critical Transition Layer
To locate where the model transitions across languages, we tracked the top-10 predicted token distribution across residual layers using the logit lens approach[cite: 5]:
$$\hat{y}_l = \text{Softmax}(W_u \cdot h_l)$$
where $h_l$ is the hidden state at layer $l$, and $W_u$ is the unembedding matrix[cite: 5].

| Layer | Top Output Behavior | Observation[cite: 5, 6] |
| :--- | :--- | :--- |
| **Layer 2** | `<em>`, `<i>`, `<strong>`, `<u>` | Syntactic/formatting tokens[cite: 5] |
| **Layer 22** | `"love"`, `"Love"`, `"amour"`, `"liefde"` | Multilingual concepts, predominantly English/Latin[cite: 5] |
| **Layer 23** | **`"αγάπη"`**, Greek tokens | **Sharp transition to Greek conceptual representation[cite: 5, 6]** |

### 2. SAE Decomposition (GemmaScope)
We loaded the pretrained SAE `layer_23/width_16k/canonical`[cite: 6]:
$$x \approx W_{dec} \cdot h + b_{dec}, \quad \text{s.t. } h \text{ is sparse}$$

The top activated latent features extracted at Layer 23[cite: 6]:
* **Feature #8189** (Activation: 86.11)[cite: 6]
* **Feature #8057** (Activation: 77.81) — *Identified & verified on Neuronpedia as the "Greek language / Greek concepts" feature[cite: 6].*
* **Feature #6504** (Activation: 72.76)[cite: 6]

### 3. Representation Steering
By clamping / adding a scaled feature vector $\alpha \cdot v_{8057}$ to the residual stream activations at Layer 23 during generation, the model shifts its completions toward Greek topics without explicit instruction[cite: 1, 6]:

* **Prompt:** `"My favorite non english word is"`[cite: 6]
  * **Base Gemma-2B:** `"sake"`, `"sherry"`, French words[cite: 6]
  * **Steered ($\alpha = 145$):** *"My favorite non english word is the Greek word γιατί, which means why..."*[cite: 6]
  * **Steered ($\alpha = 166$):** *"My favorite non english word is κλάς, which means class..."*[cite: 6]

---

## 📂 Repository Structure

```plaintext
├── notebooks/
│   └── gemma_2b_mechanistic_interpretability.ipynb   # Full interactive notebook
├── docs/
│   └── mechanistic_interpretability_paper.pdf        # Research report / case study
├── assets/
│   └── layer_activations.png                         # Visualizations & plots[cite: 6]
├── requirements.txt
├── LICENSE
└── README.md
🚀 Getting StartedPrerequisitesPython 3.10+CUDA-capable GPU (recommended for local execution of Gemma-2B)InstallationBashgit clone [https://github.com/ArisPagonopoulos/gemma-mechanistic-interpretability-greek.git](https://github.com/ArisPagonopoulos/gemma-mechanistic-interpretability-greek.git)
cd gemma-mechanistic-interpretability-greek
pip install -r requirements.txt
Dependencies (requirements.txt)Plaintexttorch>=2.0.0
transformers>=4.40.0
accelerate>=0.28.0
numpy
scipy
matplotlib
huggingface_hub
📖 References & AcknowledgmentsGemma 2: Google DeepMind (arXiv:2408.00118)   GemmaScope: Open Sparse Autoencoders on Gemma 2 (Lieberum et al., 2024)   Neuronpedia: Interactive reference and tooling for analyzing neural networks (Neuronpedia)   Towards Monosemanticity: Decomposing Language Models with Dictionary Learning (Bricken et al., Anthropic 2023)   👤 AuthorAristeidis Pagonopoulos – GitHub | LinkedIn   Department of Informatics, Aristotle University of Thessaloniki (AUTH)[cite: 1]
