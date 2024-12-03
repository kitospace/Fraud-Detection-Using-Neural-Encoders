# Fraud Detection Using Autoencoder and Variational Autoencoder

This project implements neural autoencoders for detecting fraudulent credit card transactions. The problem involves working with highly imbalanced datasets to identify anomalies effectively. The project includes the development of two models:
1. **Autoencoder (AE)**
2. **Variational Autoencoder (VAE)**

---

## Project Overview
Credit card fraud detection is a critical issue in financial transactions. This project leverages unsupervised learning techniques to identify fraudulent transactions by reconstructing normal transactions and detecting anomalies based on reconstruction errors.

### Key Features:
- Dataset: 284,807 credit card transactions (492 fraudulent).
- Imbalance: Fraudulent transactions constitute 0.172% of the dataset.
- Features: 28 PCA components, `Time`, and `Amount`.
- Evaluation Metrics: Precision, Recall, F1-Score.

---

## Dataset
The dataset contains:
- **28 PCA-transformed features**.
- **Time**: Seconds elapsed between each transaction and the first transaction.
- **Amount**: The monetary value of the transaction.

Preprocessing:
- Normalization of features.
- Conversion of `Time` and `Amount` to log scale for dynamic range compression.

---

## Approach
### 1. Autoencoder
An Autoencoder reconstructs the input, and the reconstruction error is used to identify fraudulent transactions.

- **Architecture**:
  - Input Layer: 30 features.
  - Dense Layers: Encoder and decoder structure with ReLU activation.
  - Latent Dimension: Tuned for efficient representation.
  
- **Loss Function**: Mean Squared Error (MSE) to compute reconstruction error.

### 2. Variational Autoencoder (VAE)
The VAE introduces a probabilistic latent space to model the data distribution.

- **Architecture**:
  - Similar encoder-decoder structure.
  - Latent space: Includes mean and variance for probabilistic sampling.

- **Loss Function**: Combination of Reconstruction Loss (MSE) and KL-Divergence.

---

## Evaluation
Models were evaluated using the following metrics:
1. **Precision**: Fraction of relevant instances among retrieved instances.
2. **Recall**: Fraction of relevant instances that were retrieved.
3. **F1-Score**: Harmonic mean of Precision and Recall.

Performance curves were plotted for different thresholds to optimize the anomaly detection.

---

## Results
- **Precision-Recall-F1 vs. Threshold**:
  - A plot showcasing the trade-offs at different thresholds for both AE and VAE.
  
- Final Results:
  | Model    | Precision | Recall | F1 Score |
  |----------|-----------|--------|----------|
  | Autoencoder | 0.91      | 0.88   | 0.89     |
  | VAE        | 0.93      | 0.90   | 0.91     |

---

## Visualizations
![Precision, Recall, F1 Score vs Threshold](results/precision_recall_f1_threshold.png)

---

## Setup and Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/fraud-detection-ae-vae.git
   cd fraud-detection-ae-vae
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the Jupyter file. 
