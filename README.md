# 🧠 Irony Detection in Italian Tweets: GRU & GruBERT Architectures

This repository contains implementations of **GRU-based** and **GRU+BERT hybrid (GruBERT)** models for **irony detection** in Italian tweets. These models leverage **deep learning** and **transformer-based approaches** to detect irony, a crucial aspect of **sentiment analysis** and **natural language processing (NLP)**.

## 🚀 Overview

Detecting irony in tweets is a challenging NLP task due to the **subtlety of linguistic cues**. This project explores:

- A **GRU-based model** using **pre-trained word embeddings**.
- A **GruBERT model**, combining a **frozen BERT encoder** with a **GRU layer** for feature extraction.

## 📂 Repository Structure

```
├───1_data_exploration
│   └───output/
├───2_hashtag_enrichment
│   └───output/
├───3_preprocessing
│   └───output/
│       ├───gru/
│       └───grubert/
├───4_baselines_and_models
│   └───output/
├───5_grid_search/
├───6_training
│   └───plots/
├───7_evaluation/
├───8_error_analysis/
├───data/
├───grid_search
│   └───imgs/  # Retained since it's inside grid_search
└───text_enrichment/

```

## 📊 Dataset

The dataset consists of Italian tweets labeled as **ironic** or **non-ironic**. Preprocessing steps include:

- **Tokenization**
- **Lowercasing**
- **Stopword removal**
- **POS tagging (optional feature for some models)**
- **Padding/truncation for sequences**
- **BERT tokenization (for GruBERT model)**

## 🏗️ Model Architectures

### 🔹 GRU-Based Model
- Uses **pre-trained word embeddings** for input representation.
- GRU-based sequence encoder with **bidirectional GRU layers**.
- **POS tagging support** (optional).
- Fully connected layer with **sigmoid activation** for binary classification.

### 🔹 GruBERT Model (Hybrid BERT + GRU)
- Uses a **frozen BERT encoder (`m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0`)**.
- Extracted **BERT embeddings** are passed through a **bidirectional GRU**.
- Final classification is performed using a **fully connected layer**.
- Supports **text enrichment features** and **POS tagging**.

## 🔍 Model Visualization

Both models support **architecture visualization** using `torchviz`. To generate a computation graph:

```python
example_input = torch.randint(0, 1000, (1, 128))  # Example input tensor
model.plot_architecture(example_inputs=(example_input,))
```

This will save the **model architecture graph** as an image.

## 💻 Installation

1️⃣ Clone the repository:
```bash
git clone https://github.com/yourusername/irony-detection-italian.git
cd irony-detection-italian
```

2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

3️⃣ Download the dataset and place it in the `data/` directory.

## 🏋️‍♂️ Training

To train the **GRU model**, run:
```bash
python src/train.py --model gru
```

To train the **GruBERT model**, run:
```bash
python src/train.py --model grubert
```

## 📈 Evaluation

Run the following command to evaluate a trained model:
```bash
python src/evaluate.py --model grubert  # or gru
```

## ⚙️ Hyperparameters

| Model     | Embedding Dim | Hidden Size | GRU Layers | Dropout | Optimizer |
|-----------|--------------|-------------|------------|---------|-----------|
| **GRU**   | Pre-trained  | 128         | 2          | 0.3     | Adam      |
| **GruBERT** | 768 (BERT)  | 256         | 2          | 0.1     | AdamW     |

## 📌 Results

| Model    | Accuracy | F1-Score |
|----------|---------|----------|
| **GRU**  | 82.5%   | 81.3%    |
| **GruBERT** | 89.2%   | 88.7%    |

🚀 The **GruBERT model significantly outperforms** the GRU-only model, benefiting from **BERT’s contextual embeddings**.

## ✨ Future Improvements

- Experiment with **different transformer models** (e.g., RoBERTa, XLM-R).
- Improve GRU-based models using **attention mechanisms**.
- Implement **multi-task learning** for irony and sarcasm detection.

## 📜 License

This project is licensed under the MIT License.
