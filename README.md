# Mechanistic Interpretability on Gemma-2B: Greek Language Feature & Steering

Course project for the Department of Informatics at Aristotle University of Thessaloniki (AUTH).

Authors: Aristeidis Pagonopoulos, Thomas Nektarios Saltzis, Nick Tzavidas

---

### What this is

We looked into the internal representations of Google's Gemma-2-2B to see where and how Greek language features emerge, and whether we can manipulate them. 

Instead of treating the model as a black box, we used a Sparse Autoencoder (SAE) from GemmaScope to decompose polysemantic activations into distinct, interpretable features.

### What we did

1. **Found the layer where the language flips:** Using the logit lens on a prompt that naturally completes in Greek (`the word love in greek letters is <strong> <em>`), we observed intermediate token predictions across the residual stream. The transition from general/English tokens to Greek happens sharply between **Layer 22 and Layer 23**.
2. **Extracted features with GemmaScope:** We loaded the pre-trained 16k canonical SAE for Layer 23 (`layer_23/width_16k/canonical`). The second strongest activation was **Feature 8057**, which corresponds to Greek concepts and vocabulary (verified on Neuronpedia).
3. **Activation steering:** By adding this feature vector directly into the residual stream at Layer 23 during inference, we steered the base model to start talking about Greek words (e.g., answering *"γιατί"* or *"κλάς"* to open-ended prompts like *"My favorite non english word is"*), without any explicit prompt engineering.

### Stack

- `torch`, `transformers`
- `transformer-lens` (for caching and inspecting activations)
- `sae-lens` (for loading GemmaScope SAEs)
- Model: `google/gemma-2-2b`
- SAE: `gemma-scope-2b-pt-res-canonical` (layer 23, width 16k)

### Repo Structure

- `final_mechanistic.ipynb`: The Colab/Jupyter notebook containing the full workflow (model loading, logit lens, SAE feature inspection, and steering experiments).
- `paper.pdf`: Our project report with methodology and detailed output tables.

### Quick Start

```bash
pip install -r requirements.txt
