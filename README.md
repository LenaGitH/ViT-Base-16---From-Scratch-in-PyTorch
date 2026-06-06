# Vision Transformer (ViT-Base/16) — Built From Scratch

A from-scratch PyTorch implementation of the Vision Transformer, replicating **ViT-Base/16** from the paper [*An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2020).

Rather than importing a pre-built model, every component is implemented directly from the paper's four core equations.

## Why this project

The goal was not to beat a benchmark, but to *understand* the architecture by rebuilding it piece by piece — and to observe first-hand why Vision Transformers behave the way they do.

## Architecture

The model is built as composable `nn.Module` blocks, mapped to the paper's equations:

| Component | Paper | What it does |
|---|---|---|
| **Patch Embedding** | Eq. 1 | Splits the image into 16×16 patches with a strided `Conv2d`, prepends a learnable class token, and adds learnable position embeddings |
| **MSA Block** | Eq. 2 | LayerNorm → Multi-Head Self-Attention, with a residual connection |
| **MLP Block** | Eq. 3 | LayerNorm → MLP (Linear → GELU → Linear), with a residual connection |
| **Transformer Encoder** | Eq. 2 + 3 | Stacks the MSA and MLP blocks ×12 |
| **Classifier Head** | Eq. 4 | LayerNorm → Linear, reading from the class token |

### Model specs

| Hyperparameter | Value |
|---|---|
| Patch size | 16 |
| Embedding dim (D) | 768 |
| Layers | 12 |
| Attention heads | 12 |
| MLP size | 3072 |
| Parameters | ~86M (verified: 85,806,346) |

## Dataset

CIFAR-10, resized to 224×224 to match the ViT-Base/16 input configuration.

## Results & key takeaway

Trained from scratch, the model **learns** — test accuracy rises above the 10% random baseline — but stays well below what a comparable CNN achieves on the same data.

This is the central lesson, not a shortcoming of the implementation: Vision Transformers lack the spatial inductive biases that CNNs have built in, which makes them **data-hungry**. The original ViT was pre-trained on ~300M images (JFT-300M) before fine-tuning. On a small dataset and without pre-training, underperformance is expected — and reproducing it directly made the trade-off concrete.

## Engineering note

A parameter-count sanity check (expected ~86M) caught a silent bug where the batch size had been accidentally baked into the class-token and position-embedding parameters — a mismatch that output-shape checks alone could not reveal.

## References

- Dosovitskiy et al. (2020), *An Image is Worth 16×16 Words* — [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Built while following the [learnpytorch.io](https://www.learnpytorch.io/) curriculum
