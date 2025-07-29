# 📧 Email Classification with BERT

An advanced **natural language processing** solution for email spam detection using **BERT transformers**, featuring **DistilBERT architecture**, **cross-validation**, **robust preprocessing**, and **production-ready inference**. This project implements state-of-the-art transformer models for accurate email classification with comprehensive evaluation metrics.

## 📋 Table of Contents

1. [About the Dataset](#about-the-dataset)  
2. [Project Structure](#project-structure)  
3. [Model Architecture](#model-architecture)  
4. [Key Features](#key-features)  
5. [Training Pipeline Overview](#training-pipeline-overview)  
6. [Requirements](#requirements)  
7. [Results](#results)  
8. [License](#license)
9. [Contributing](#contributing)

## 📊 About the Dataset

This system is built for **Email Classification**:

- **Email text data** with corresponding **spam/ham labels**  
- **Binary classification**: **0 (Ham)** for legitimate emails, **1 (Spam)** for unwanted emails  
- **CSV format** with `text` and `spam` columns  
- **Automatic preprocessing** including text cleaning and length optimization  
- **Stratified sampling** ensures balanced training across classes  

## 🗂 Project Structure

```
Email-Spam-Classification/
├── .gitignore
├── requirements.txt
├── LICENSE
├── README.md                       # Project documentation (this file)
├── emails.csv                      # Input email dataset
├── spam_classifier.pth             # Trained model checkpoint, available in releases.
└──  email_bert_classifier.ipynb    # Complete implementation
```

## 🧠 Model Architecture

```
Email Text Input
        ↓
DistilBERT Tokenizer (Max Length: 128/256)
        ↓
DistilBERT Encoder (768-dimensional embeddings)
        ↓
[CLS] Token Extraction → Dropout (0.3)
        ↓
Linear (768 → 384) → ReLU → Dropout (0.3)
        ↓
Linear (384 → 2) → Softmax
        ↓
Output: [Ham Probability, Spam Probability]
```


## 🧠 Model Overview

- **Transformer Backbone**: **DistilBERT-base-uncased** for efficient text understanding  
- **Custom Classifier Head**: Two-layer neural network with dropout regularization  
- **Binary Classification**: Outputs probability scores for ham/spam classification  

## ✨ Key Features

- **Advanced Tokenization**: BERT WordPiece tokenization with attention masks  
- **Cross-Validation**: **5-fold stratified** validation for robust model evaluation  
- **Differential Learning Rates**: Separate rates for BERT layers (1e-5) and classifier (2e-5)  
- **Gradient Clipping**: Prevents exploding gradients with max norm 1.0  
- **Label Smoothing**: Optional cross-entropy enhancement for better calibration  
- **Model Persistence**: Save/load functionality for production deployment  
- **Comprehensive Metrics**: Accuracy, F1-score, precision, recall, and confusion matrices  

## 🔁 Training Pipeline Overview

### 1. 📂 Data Preparation
- **Text cleaning** and normalization with whitespace handling  
- **Adaptive max length** based on 95th percentile of text lengths  
- **Stratified splitting** maintains class balance across folds  
- **Tokenization** with truncation and padding for uniform input size  

### 2. ⚙️ Model Setup
- **DistilBERT** pre-trained weights with custom classification head  
- **Xavier initialization** for linear layers  
- **Dropout regularization** (0.3) in both hidden and output layers  

### 3. 🏋️ Training
- **AdamW optimizer** with weight decay (0.01) for regularization  
- **Linear warmup scheduler** with 10% warmup steps  
- **Early stopping** based on validation accuracy improvement  
- **Cross-validation** ensures model generalization across data splits  

### 4. 🧪 Evaluation
- **Per-fold accuracy** tracking with statistical analysis  
- **Classification reports** with precision/recall per class  
- **Confusion matrix** visualization for error analysis  
- **Test set evaluation** on held-out data for final performance  

## ⚙️ Requirements

```bash
pandas>=1.5.0
numpy>=1.21.0
torch>=1.12.0  
transformers>=4.20.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
tqdm>=4.64.0
```

These requirements can be easily installed by: `pip install -r requirements.txt`


## 📈 Results

- **Cross-Validation Accuracy**: **97.56% ± 0.87%** across 5 folds  
- **F1-Score**: **97.51%** weighted average for balanced performance  
- **Test Accuracy**: Consistently exceeds **95%** on held-out data  
- **Inference Speed**: **100-500 emails/second** on GPU, **50-100 emails/second** on CPU  
- **Model Size**: **66MB** compressed checkpoint for efficient deployment  

### Performance Breakdown:
- **Ham (Legitimate) Precision**: ~98%  
- **Spam Detection Recall**: ~96%  
- **False Positive Rate**: <2%  
- **False Negative Rate**: <4%  

## 📄 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

### 💡 Opportunities for Contribution:

- **Model Variants**: Experiment with **RoBERTa**, **ALBERT**, or **ELECTRA** architectures  
- **Advanced Preprocessing**: Implement **HTML tag removal**, **URL normalization**, and **language detection**  
- **Ensemble Methods**: Combine multiple transformer models for improved accuracy  
- **Real-time API**: Build **FastAPI** or **Flask** endpoints for production serving  
- **MLOps Integration**: Add **MLflow** tracking, **Docker** containerization, and **CI/CD** pipelines  

### 🔧 How to Contribute:

1. Fork the repository  
2. Create a feature branch  
```bash
git checkout -b feature/new-enhancement
```
3. Implement changes with comprehensive testing and documentation  
4. Submit a pull request with detailed description and performance benchmarks  


⭐ If this project helps you build better email classification systems, consider giving it a star!



