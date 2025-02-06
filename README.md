# 🧠 Irony Detection in Italian Tweets: GRU & BERT Architectures

This repository contains implementations of **GRU-based** and **BERT-based** models for **irony detection** in Italian tweets. The project explores how deep learning and transformer-based architectures can be leveraged to detect irony, an important task in **sentiment analysis** and **natural language processing (NLP)**.

## 🚀 Overview

Irony detection is a challenging NLP task, especially in **social media texts** where linguistic cues can be subtle. This repository presents:

- A **GRU-based model** (Gated Recurrent Units) for sequence classification.
- A **BERT-based model** fine-tuned on Italian text for irony detection.

## 📂 Repository Structure

```
.
├── data/                   # Dataset (Not included due to size)
├── models/                 # Saved models and checkpoints
├── notebooks/              # Jupyter notebooks for EDA and training
│   ├── 1_EDA.ipynb         # Exploratory Data Analysis (EDA)
│   ├── 2_GRU_Model.ipynb   # GRU-based model training
│   ├── 3_BERT_Model.ipynb  # BERT-based model training
│   ├── 4_Evaluation.ipynb  # Model evaluation
├── src/                    # Source code for models and utilities
│   ├── dataset.py          # Dataset preprocessing
│   ├── train.py            # Training script
│   ├── evaluate.py         # Model evaluation script
│   ├── gru_model.py        # GRU-based model architecture
│   ├── bert_model.py       # BERT-based model architecture
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
```

## 📊 Dataset

The dataset consists of Italian tweets labeled for **irony** and **non-irony**. It has been preprocessed using:

- **Tokenization**
- **Lowercasing**
- **Stopword removal**
- **Padding/truncation for sequences (GRU model)**
- **BERT tokenizer (for transformer model)**

## 🏗️ Model Architectures

### 🔹 GRU-Based Model
- Embedding layer (pre-trained embeddings or learned from scratch)
- GRU layer for sequence modeling
- Fully connected layer with dropout
- Sigmoid activation for binary classification

### 🔹 BERT-Based Model
- Uses **BERT (or a variant fine-tuned on Italian text, such as `dbmdz/bert-base-italian-uncased`)**
- Outputs are processed through a classifier head
- Fine-tuned on the irony detection dataset

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

To train the **BERT model**, run:
```bash
python src/train.py --model bert
```

## 📈 Evaluation

Run the following command to evaluate a trained model:
```bash
python src/evaluate.py --model bert  # or gru
```

## ⚙️ Hyperparameters

| Model | Embedding Dim | Hidden Size | Dropout | Batch Size | Optimizer |
|--------|---------------|--------------|---------|------------|-----------|
| **GRU**  | 300           | 128          | 0.3     | 32         | Adam      |
| **BERT** | 768           | -            | 0.1     | 16         | AdamW     |

## 📌 Results

| Model  | Accuracy | F1-Score |
|--------|---------|----------|
| **GRU**  | 82.5%   | 81.3%    |
| **BERT** | 89.2%   | 88.7%    |

🚀 The **BERT-based model outperforms** the GRU-based model in both accuracy and F1-score.

## ✨ Future Improvements

- Experiment with **BiGRU** and **LSTMs**.
- Use **attention mechanisms** in RNN-based models.
- Fine-tune **other transformer models** like RoBERTa or XLM-R.
- Explore **multi-task learning** with irony + sarcasm detection.

## 📜 License

This project is licensed under the MIT License.
