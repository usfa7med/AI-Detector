# 🔍 AI Detector

### Training a Custom ML Model on ~300k Samples to Detect AI-Generated Text

![Language](https://img.shields.io/badge/Language-Python-blue?style=for-the-badge&logo=python&logoColor=white)
![ML](https://img.shields.io/badge/ML-Logistic%20Regression-purple?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-~300k%20samples-red?style=for-the-badge)
![Features](https://img.shields.io/badge/Features-29%20NLP-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-Educational-red?style=for-the-badge)

AI Detector is a custom machine learning system that classifies text as human-written or AI-generated. A **Logistic Regression classifier** was trained from scratch on **~299,000 labeled samples** using **29 handcrafted NLP features** interpretable signals like Shannon entropy, Yule's K, MATTR, Flesch readability, POS ratios, n-gram repetition, and cosine sentence coherence.

Unlike most "AI detectors" that wrap third-party APIs, this project trains a model that looks at measurable, interpretable signals in the text not "asking another AI."

**Live demo:** [ai-detector.usfahmed.dev](https://ai-detector.usfahmed.dev)

---

## 📑 Table of Contents

- [🎯 Training Pipeline](#-training-pipeline)
- [📊 Dataset](#-dataset)
- [🔬 Feature Engineering](#-feature-engineering)
- [🧠 Model Architecture](#-model-architecture)
- [⚙️ Installation](#%EF%B8%8F-installation)
- [🚀 Usage](#-usage)
- [📂 Project Structure](#-project-structure)
- [⚠️ Challenges & Solutions](#%EF%B8%8F-challenges--solutions)
- [🔮 Future Improvements](#-future-improvements)
- [🛠 Technologies Used](#-technologies-used)
- [📄 License](#-license)
- [⚠️ Disclaimer](#-disclaimer)
- [📫 Contact](#-contact)

---

## 🎯 Training Pipeline

The core of this project is the offline training pipeline that produces the model.

```text
┌──────────────────────────────────────────────────────────────────┐
│                    Training Pipeline (Python / Kaggle)            │
│                                                                   │
│  ┌──────────────┐     ┌──────────────────┐     ┌──────────────┐ │
│  │  ~300k Text   │────▶│  Feature Extract  │────▶│  29 Features │ │
│  │  Samples      │     │  (per-sample)     │     │  (per row)   │ │
│  └──────────────┘     └──────────────────┘     └──────┬───────┘ │
│                                                        │          │
│                                                        ▼          │
│  ┌──────────────────┐     ┌──────────────────┐                   │
│  │  model.json       │◀────│  Logistic Reg.   │                   │
│  │  (29 weights)     │     │  (from scratch)  │                   │
│  └──────────────────┘     └──────────────────┘                   │
└──────────────────────────────────────────────────────────────────┘
```

1. **Load & clean** ~300k rows with a robust CSV parser (handles quoted text with commas, encoding errors)
2. **Extract 29 features** per sample
3. **Train** Logistic Regression from scratch with balanced class weights and L2 regularization
4. **Export** weights + bias + z-score normalization params to a 5KB JSON file

---

## 📊 Dataset

- **~299,000 total samples** (150,000 AI-generated, 149,128 human-written)
- Split across three CSV files: `ai_data.csv`, `human_data.csv`, `full_data.csv`
- **95/5 train/test split** with fixed seed (42) for reproducibility
- ~284,000 training samples, ~15,000 test samples

Download from Google Drive: [**AI Detector Datasets**](https://drive.google.com/drive/folders/1p7mFJcsBUODKe4-jp1-cf8HiJ9shccna?usp=sharing)

Place the downloaded folders so the structure looks like:
```text
AI Detector/
├── datasets/
│   ├── ai_data.csv        # ~150k AI-generated samples
│   ├── human_data.csv     # ~149k human-written samples
│   └── full_data.csv      # Combined ~299k samples
└── training/
    └── extracted_features.csv
```

---

## 🔬 Feature Engineering

The model uses **29 handcrafted NLP features** across four families no black-box embeddings, no transformer internals, every signal is named and auditable.

### Lexical Diversity
| Feature | Description |
|---------|-------------|
| Shannon Entropy | Entropy of the word frequency distribution AI text tends toward lower entropy |
| Lexical Density | Ratio of content words to total words |
| MATTR | Moving Average Type-Token Ratio over 20-word windows |
| Top Word Freq Ratio | Frequency of the most common word higher in AI text |

### Syntactic
| Feature | Description |
|---------|-------------|
| Verb Ratio | Proportion of words classified as verbs (suffix + dictionary) |
| Noun Ratio | Proportion of words classified as nouns |
| Adjective Ratio | Proportion of adjectives |
| Adverb Ratio | Proportion of adverbs |
| Passive Voice Ratio | Sentences with auxiliary + past-participle pattern |
| Conjunction Ratio | Frequency of coordinating/subordinating conjunctions |

### Structural
| Feature | Description |
|---------|-------------|
| Avg Word Length | Mean characters per word |
| Avg Sentence Length (words) | Mean words per sentence |
| Avg Sentence Length (chars) | Mean characters per sentence |
| Sentence Length StdDev | Variability in sentence length |
| Top 10% vs Rest Ratio | Longest sentences vs average AI text is more uniform |
| Sentence Length CV | Coefficient of variation of sentence lengths |
| Comma Ratio | Commas per word |
| Punctuation Ratio | Total punctuation per word |
| Yule's K | Vocabulary richness metric lower K = more even word usage (typical of AI) |

### Discourse
| Feature | Description |
|---------|-------------|
| Coherence Mean | Mean cosine similarity between adjacent sentence BoW vectors |
| Coherence Variance | Variance of pairwise cosine similarity lower in AI text |
| Bigram Repetition | Fraction of repeated bigrams |
| Trigram Repetition | Fraction of repeated trigrams |
| Top Trigram Ratio | Fraction of the most frequent trigram higher in formulaic AI text |
| Stopword Ratio | Frequency of stopwords |
| Flesch Reading Ease | Readability score |
| Structural Parallelism | Fraction of sentences with matching opening word + length bucket |

---

## 🧠 Model Architecture

**Logistic Regression from scratch** no scikit-learn at inference, no TensorFlow.js, no ONNX runtime.

### Training

```python
model = LogisticRegressionScratch(n_features=29)
model.fit(
    X_train, y_train,
    epochs=3000, lr=0.1,
    l2_lambda=0.05,
    class_weight="balanced",
)
model.export_weights("model.json")
```

- **95/5 train/test split** with fixed seed (42) for reproducibility
- **Balanced class weights** prevent majority-class bias (150k AI vs 149k human)
- **L2 regularization** (λ=0.05) prevents overfitting to training-specific patterns
- **Z-score normalization** (mean/std from training set) serialized into model.json

### Inference

```typescript
export function predictProba(features: number[], model: ModelWeights): number {
  let z = model.b;
  for (let i = 0; i < features.length; i++) {
    let std = model.std[i];
    if (std === 0) std = 1.0;
    const scaledFeature = (features[i] - model.mean[i]) / std;
    z += model.w[i] * scaledFeature;
  }
  return sigmoid(z);
}
```

- **29 multiplications + 29 additions + 1 sigmoid** pure arithmetic, no branches
- **model.json** is under 5KB (29 weights + bias + 29 means + 29 stds)
- **Zero npm dependencies** at inference time

---

## ⚙️ Installation

### Prerequisites

- [Python 3.10+](https://python.org/)
- [Node.js](https://nodejs.org/) v18+ (for TypeScript inference port)

### Setup

```bash
git clone https://github.com/usfa7med/ai-detector.git
cd ai-detector
```

**Training:**
```bash
cd training
pip install -r requirements.txt
```

**Datasets (large files hosted externally):**

📥 [**AI Detector Datasets & Training Data**](https://drive.google.com/drive/folders/1p7mFJcsBUODKe4-jp1-cf8HiJ9shccna?usp=sharing)

---

## 🚀 Usage

### Training the Model

```bash
cd training
python train.py
```

This will:
1. Load ~300k rows from `datasets/full_data.csv`
2. Extract features (resumable picks up where it left off if interrupted)
3. Train Logistic Regression for 3,000 epochs
4. Export `model.json` with weights, bias, and normalization params

### Loading the Trained Model

```python
import json

with open("model.json") as f:
    model = json.load(f)

# model["w"]  29 weights
# model["b"]  bias
# model["mean"] training set means (for z-score normalization)
# model["std"]  training set stds (for z-score normalization)
```

### Making Predictions

```python
import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def predict(text_features, model):
    x = np.array(text_features)
    x_scaled = (x - np.array(model["mean"])) / np.array(model["std"])
    z = np.dot(x_scaled, model["w"]) + model["b"]
    probability = sigmoid(z)
    label = 1 if probability >= 0.5 else 0
    return {"probability": float(probability), "label": label, "prediction": "ai" if label == 1 else "human"}
```

---

## 📂 Project Structure

```text
AI Detector/
├── training/                  # Offline model training (Python)
│   ├── train.py               # Full pipeline: features → LR → export
│   ├── extracted_features.csv  # (downloaded from Google Drive)
│   └── requirements.txt
│
├── datasets/                  # Training data (downloaded from Google Drive)
│   ├── .gitkeep               # Directory placeholder
│   ├── ai_data.csv            # ~150k AI-generated samples (ignored in git)
│   ├── human_data.csv         # ~149k human-written samples (ignored in git)
│   └── full_data.csv          # Combined ~299k samples (ignored in git)
│
└── model.json                 # Trained weights + bias + z-score normalization (5KB)
```

> **Note:** The web frontend and TypeScript inference code have been omitted from this repository. This repository focuses solely on the machine learning training pipeline.

---

## ⚠️ Challenges & Solutions

### Python → TypeScript Feature Parity

**Problem:** The Python training pipeline uses `string.punctuation`, numpy, and sklearn, but the TypeScript inference version must reimplement everything without dependencies.

**Solution:** Built a zero-dependency TypeScript feature extractor (~250 lines) that mirrors the Python pipeline exactly same sentence splitter regex, same syllable counter, same Yule's K formula, same cosine similarity via manual bag-of-words vectors. Verified feature output matches between Python and TypeScript for the same input text.

---

### Resumable Feature Extraction at Scale

**Problem:** Extracting 29 features across 300k texts takes 6-8 hours on a Kaggle GPU. Kaggle kernels have a 12-hour time limit.

**Solution:** Built an incremental-write pipeline that flushes every 100 rows to disk and resumes from the last checkpoint on restart. A robust CSV parser handles quoted text with commas and encoding errors via regex-based record boundary detection.

---

### Balanced Class Weights

**Problem:** The dataset is nearly balanced (150k AI vs 149k human), but even a slight imbalance can bias the model toward the majority class.

**Solution:** Used `class_weight="balanced"` in the custom LogisticRegressionScratch, which computes per-class weights as `n_samples / (n_classes * n_samples_per_class)`. After normalization, both classes contribute equally to the loss function during training.

---

## 🔮 Future Improvements

- Add **sentence highlighting** show which sentences pushed the prediction toward AI
- Support **multi-language detection** with language-specific feature sets
- Add **batch analysis** for checking multiple documents at once
- Implement a **feedback loop** where users can flag incorrect predictions to retrain the model
- Explore **ensemble methods** combining logistic regression with a lightweight transformer
- Train on **larger, more diverse datasets** to improve generalization across AI generators

---

## 🛠 Technologies Used

### Machine Learning
- **Logistic Regression** (from scratch) Binary classifier, 29 features, L2 regularization
- **scikit-learn** Training-time feature extraction (CountVectorizer, cosine similarity)
- **pandas / NumPy** Data loading + numerical computation

### Inference
- **TypeScript** Zero-dependency feature extractor + inference engine

---

## 📄 License

This project is licensed under a **Custom Educational License** all rights reserved.

See the [`LICENSE`](LICENSE) file for full details.

---

## ⚠️ Disclaimer

AI Detector is an **educational and research tool**. No AI detector is 100% accurate. Results should be treated as a **strong signal, not definitive proof** especially for high-stakes decisions such as academic integrity cases. Very short texts, heavily edited AI text, or human text that mimics AI patterns can all reduce accuracy. Use responsibly.

---

# 📫 Contact

**Youssef Ahmed Abdelfatah**

🌐 **Portfolio**
https://usfahmed.dev

💻 **GitHub**
https://github.com/usfa7med

💼 **LinkedIn**
https://linkedin.com/in/usfahmed

✉️ **Email**
hello@usfahmed.dev
