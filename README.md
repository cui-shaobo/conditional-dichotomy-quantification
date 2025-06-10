# Dichotomous Score
*A lightweight toolkit for measuring how “opposite” two texts are when they share the same context.*

<a href="https://pypi.org/project/opposite-score/"> <img src="https://img.shields.io/pypi/v/opposite-score?style=flat-square" alt="PyPI version" /> </a>


<a href="https://huggingface.co/shaobocui/opposite-score-debate-bert"> <img src="https://img.shields.io/badge/Hugging%20Face-opposite--score--debate--bert-yellow?logo=huggingface&style=flat-square" alt="debate-bert" /> </a> <a href="https://huggingface.co/shaobocui/opposite-score-debate-roberta"> <img src="https://img.shields.io/badge/Hugging%20Face-opposite--score--debate--roberta-yellow?logo=huggingface&style=flat-square" alt="debate-roberta" /> </a> <a href="https://huggingface.co/shaobocui/opposite-score-defeasibleNLI-bert"> <img src="https://img.shields.io/badge/Hugging%20Face-opposite--score--defeasibleNLI--bert-yellow?logo=huggingface&style=flat-square" alt="defeasibleNLI-bert" /> </a> <a href="https://huggingface.co/shaobocui/opposite-score-defeasibleNLI-roberta"> <img src="https://img.shields.io/badge/Hugging%20Face-opposite--score--defeasibleNLI--roberta-yellow?logo=huggingface&style=flat-square" alt="defeasibleNLI-roberta" /> </a> <a href="https://huggingface.co/shaobocui/opposite-score-causal-reasoning-bert"> <img src="https://img.shields.io/badge/Hugging%20Face-opposite--score--causal--reasoning--bert-yellow?logo=huggingface&style=flat-square" alt="causal-reasoning-bert" /> </a> <a href="https://huggingface.co/shaobocui/opposite-score-causal-reasoning-roberta"> <img src="https://img.shields.io/badge/Hugging%20Face-opposite--score--causal--reasoning--roberta-yellow?logo=huggingface&style=flat-square" alt="causal-reasoning-roberta" /> </a>

---
## 📚 Table of Contents

- [✨ Why should I care?](#-why-should-i-care)
- [📦 Installation](#-installation)
- [🗂️ Supportive Datasets (Quick Overview)](#️-supportive-datasets-quick-overview)
- [Opposite-Score](#opposite-score)
  - [Features](#features)
  - [Installation](#installation-1)
  - [💡 Usage Examples](#-usage-examples)
- [🛡 Responsible Usage](#-responsible-usage)
- [📊 Leaderboard: Dichotomy Quantification](#-leaderboard-dichotomy-quantification)


## ✨ Why should I care?

| Domain                                                                                                                                 | One-line use-case | Why it matters |
|----------------------------------------------------------------------------------------------------------------------------------------|-------------------|----------------|
| **Public policy**<br><img src="https://github.com/cui-shaobo/public-images/raw/main/oppositescore/public_policy.png" width="280"/>     | Cluster pro ∕ con arguments from citizen consultations | Produces balanced, evidence-based draft regulations |
| **Social media**<br><img src="https://github.com/cui-shaobo/public-images/raw/main/oppositescore/social_media.png" width="280"/>       | Detect emerging polarised clusters in real time | Enables early de-escalation and healthier discourse |
| **Journalism**<br><img src="https://github.com/cui-shaobo/public-images/raw/main/oppositescore/journalism.png" width="280"/>           | Surface the strongest counter-evidence to viral claims | Speeds up balanced fact-checking & boosts information integrity |
| **Causal analysis**<br><img src="https://github.com/cui-shaobo/public-images/raw/main/oppositescore/causal_analysis.png" width="280"/> | Rank supporters vs defeaters for a suspected cause | Accelerates root-cause analysis during critical incidents |


Powered by our **Opposite-Score** embeddings and three rigorously curated datasets (Debate ▪︎ Defeasible NLI ▪︎ Causal Reasoning).

---
## 📦 Installation

```bash
conda create -n dichotomy python=3.10
conda activate dichotomy

## opposite-score is our implemented package: https://pypi.org/project/opposite-score/
pip install opposite-score
```


## 🗂️ Supportive Datasets (Quick Overview)

| Scenario | Train | Val | Test | Total | Avg len (ctx) | Avg len (pos/neg/neu) |
|----------|------:|----:|-----:|------:|--------------:|----------------------:|
| Debate               | 58 k | 21 k | 16 k | **95 k** | 8.8 | 11.6 / 11.5 / 11.2 |
| Defeasible NLI       | 8 k  | 8 k  | 424 k | **441 k** | 23.1 | 8.5 / 8.3 / 8.4 |
| Causal Reasoning     | 14 k | 18 k | 16 k | **48 k** | 21.0 | 8.4 / 10.1 / 9.1 |

![Sentence-length distributions](https://github.com/cui-shaobo/public-images/raw/main/oppositescore/sentence_length.png)

**Figure 1.** Sentence-length distributions for contexts, positive, negative, and neutral arguments across datasets.


> **Why it matters**  
> *Balanced lengths & human-verified neutrals stop models from “cheating” on superficial cues and keep the focus on genuine oppositional content.*


# Opposite-Score
Efficient embeddings and scoring mechanism for detecting contrasting or opposite relationships in text, based on a given context.

[Opposite-Score](https://pypi.org/project/opposite-score/) is designed to generate embeddings and compute the **opposite-score**, which quantifies the degree of contrast or opposition between two textual outputs within the same context. This package is particularly useful in scenarios like debates, legal reasoning, and causal analysis where contrasting perspectives need to be evaluated based on shared input.

## Features
- **Opposite-Score Calculation**: Computes a numerical score representing how opposite two texts are, conditioned on a shared context.
- **Opposite-Aware Embeddings**: Generates embeddings optimized for contrasting textual relationships.
- **Easy to Use**: Only a few lines of code to get sentence/token-level embeddings and calculate opposite scores.
- **Automatic Model Download**: The first initialization automatically downloads and installs the pre-trained Opposite-Score model.

## Installation
Install Opposite-Score via pip:
```bash
pip install opposite-score==0.0.1
```



## 💡 Usage Examples

### For Opposite Scores of two sentence conditioned on the context.

```python

from oppositescore.model.dichotomye import DichotomyE

# Example inputs
context = ["A company launches a revolutionary product."]
sentence1 = ["Competitors quickly release similar products, reducing the company's advantage."]
sentence2 = ["The company gains a significant advantage due to its unique product."]

# Initialize the model
opposite_scorer = DichotomyE.from_pretrained('shaobocui/opposite-score-debate-bert', pooling_strategy='cls').cuda()

# Calculate opposite-score (using cosine similarity as an example)
opposite_score = opposite_scorer.calculate_opposite_score(ctx=context, sent1=sentence1, sent2=sentence2)
print('Opposite Score:', opposite_score)
# Output: Opposite Score: 1.5123178
```


## 🛡 Responsible Usage

This software is released for research and educational purposes only. It is intended to support studies on argument contrast, causal reasoning, and sentence embeddings. 

Please ensure proper attribution when using the code, models, or datasets in publications or derivative work. Commercial use are expected to contact authors for explicit permission.

For questions or collaborations, feel free to contact the authors.

## 📊 Leaderboard: Dichotomy Quantification

| Model                  | Debate (DCF ↑) | Debate (Angle ↑) | NLI (DCF ↑) | NLI (Angle ↑) | Causal (DCF ↑) | Causal (Angle ↑) |
|------------------------|----------------|------------------|-------------|----------------|----------------|------------------|
| InferSent-GloVe        | 36.19          | 1.58             | 23.11       | 0.39           | 26.71          | 0.44             |
| InferSent-fastText     | 42.02          | 4.56             | 27.66       | 1.42           | 32.36          | 1.44             |
| USE                    | 16.53          | 3.31             | 18.07       | 1.01           | 13.54          | 0.46             |
| BERT                   | 31.37          | 0.18             | 11.99       | 0.25           | 27.17          | 0.26             |
| CoSENT                 | 38.49          | 0.64             | 26.86       | 0.28           | 30.07          | 0.14             |
| SBERT                  | 31.61          | 1.50             | 22.89       | 0.64           | 22.68          | 0.43             |
| SimCSE (BERT)          | 30.59          | 2.78             | 13.91       | 0.93           | 25.15          | 1.30             |
| AoE (BERT)             | 26.27          | 0.48             | 24.02       | 0.10           | 30.09          | 0.11             |
| RoBERTa                | 43.61          | 0.00             | 12.60       | 0.00           | 24.06          | 0.00             |
| SimCSE (RoBERTa)       | 30.84          | 2.42             | 12.78       | 0.64           | 27.01          | 1.28             |
| LLaMA-2 (7B)           | 30.46          | 16.99            | 21.25       | 8.65           | 32.80          | 5.67             |
| LLaMA-2 (13B)          | 47.42          | 11.24            | 30.27       | 4.59           | 34.05          | 2.56             |
| AoE (7B)               | 38.92          | 14.85            | 20.01       | 8.22           | 27.20          | 4.03             |
| AoE (13B)              | 44.88          | 9.73             | 28.72       | 3.20           | 30.89          | 1.58             |
| LLaMA-3.1 (8B)         | 39.81          | 10.86            | 21.70       | 5.33           | 26.38          | 2.67             |
| LLaMA-3.1 (70B)        | 34.47          | 13.74            | 15.95       | 6.83           | 25.23          | 3.84             |
| Ours (BERT)            | 46.97          | 30.66            | 41.72       | 3.25           | 67.59          | 20.69            |
| Ours (RoBERTa)         | **55.93**      | **83.67**        | **47.27**   | 0.63           | **76.55**      | 5.06             |


> 📌 *Both metrics benefit from higher values: better classification and stronger geometric contrast.*

📄 [Full Table with Ablations →](./docs/leaderboard_full.pdf)
