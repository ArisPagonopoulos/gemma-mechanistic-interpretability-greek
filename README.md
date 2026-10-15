# Exploring Mechanistic Interpretability: Greek Language Feature Discovery & Steering on Gemma-2B

## 1. Environment Setup & Dependencies
- Imports (PyTorch, Transformers, GemmaScope / SAE tools)
- Device configuration (CUDA/GPU setup)

## 2. Model Loading & Baseline Evaluation
- Loading `google/gemma-2-2b`
- Defining helper functions for unembedding & logit analysis

## 3. Logit Lens: Pinpointing the Language Transition Layer
- Forward pass probing with targeted prompts
- Layer-by-layer unembedding projection & top-k token probabilities (Layers 20–24)
- Identifying the semantic shift at Layer 23

## 4. Sparse Autoencoder (SAE) Activation Analysis
- Loading GemmaScope SAE (`layer_23/width_16k/canonical`)
- Extracting latent feature activations for Layer 23
- Feature verification against Neuronpedia (Feature ID: 8057)

## 5. Representation Steering & Intervention
- Hooking into the residual stream with steering vectors
- Modulating steering coefficients ($\alpha \in [145, 250]$)
- Comparative generation: Baseline Gemma-2B vs. Steered Model

## 6. Conclusions & Quantitative Findings
- Summary of findings on polysemanticity and representation steering
