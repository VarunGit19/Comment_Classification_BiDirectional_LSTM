# Comment Classification using Bidirectional LSTMs

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## Project Overview
This project tackles the real-world problem of identifying and classifying toxic online behavior. Using a Deep Learning approach (Bidirectional LSTMs), the model performs **multi-label classification** on Wikipedia comments, categorizing them into one or more of six distinct toxicity types: 
`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.

## 📊 The Dataset & The Imbalance Problem
The data is sourced from the [Kaggle Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). 

**Key Challenge:** Extreme Class Imbalance. 
The dataset consists of ~159,000 training samples, but the distribution of toxic classes is highly skewed. For example, the `threat` class makes up only **0.29%** of the data. 

Because of this, standard "Accuracy" is a deceptive metric. A model predicting "0" for everything would achieve 99% accuracy on threats while failing its actual objective. Therefore, this model is compiled and evaluated entirely on **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**, which accurately measures the model's ability to distinguish between positive and negative classes.

## Model Architecture & Engineering
The model is built using TensorFlow/Keras with the following architecture:
1. **Text Preprocessing:** Tokenization (top 20,000 words) and sequence padding to `maxlen=1400` based on EDA percentile distribution.
2. **Embedding Layer:** Transforms integer sequences into dense vectors.
3. **SpatialDropout1D (0.2):** Drops entire 1D feature maps (words) during training to prevent the network from memorizing specific words, heavily reducing overfitting on long sequences.
4. **Stacked BiLSTMs (2 Layers):** 
   * *Layer 1:* Bidirectional LSTM (64 units) returning sequences.
   * *Layer 2:* Bidirectional LSTM (64 units) returning final states.
5. **Standard Dropout:** Added between LSTMs and the final layer for regularization.
6. **Dense Output:** 6 nodes with a `sigmoid` activation function (essential for multi-label classification) paired with `binary_crossentropy` loss.

## 🛠️ Mitigating Overfitting
Given the long sequence length (1400) and the parameter-heavy stacked LSTMs, several anti-overfitting mechanisms were implemented:
* **Early Stopping:** Monitors `val_auc` and halts training if the model fails to improve for 2 epochs, automatically restoring the best weights.
* **Aggressive Dropout:** Utilizing both Spatial Dropout on embeddings and standard Dropout on recurrent layers.

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/toxic-comment-bilstm.git
   cd toxic-comment-bilstm
