# 📧 Email Classification with BERT & RoBERTa

An advanced **natural language processing** solution for email spam detection using **BERT and RoBERTa transformers**, featuring **DistilBERT architecture**, **anti-overfitting optimization**, **cross-validation**, **robust preprocessing**, and **production-ready inference**. This project implements state-of-the-art transformer models for accurate email classification with comprehensive evaluation metrics and intelligent model detection.

## 📋 Table of Contents

1. [About the Dataset](#about-the-dataset)  
2. [Project Structure](#project-structure)  
3. [Model Architectures](#model-architectures)  
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
- **Data augmentation** with word dropout and swapping techniques (RoBERTa implementation)

## 🗂 Project Structure

```
Email-Spam-Classification/
├── .gitignore
├── LICENSE
├── requirements.txt
├── emails.csv                      # Input email dataset
├── README.md                       # Project documentation (this file)
├── BERT/
│   ├── spam_classifier.pth         # Available in releases with tag 'bert'
│   └── email_bert_classifier.ipynb # Complete BERT implementation
└── RoBERTa/
    ├── optimized_roberta_spam_classifier.pth  # Available in releases with tag 'roberta'
    └── email_roBERTa_classifier.ipynb        # Complete RoBERTa implementation

```


## 🧠 Model Architectures

### 🔷 BERT Implementation


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

### 🔶 RoBERTa Implementation (Optimized)


```
Email Text Input
        ↓
RoBERTa Tokenizer (Max Length: 128)
        ↓
RoBERTa Encoder (768-dimensional embeddings)
├── Embeddings (frozen)
├── Encoder Layers 0-5 (frozen)
└── Encoder Layers 6-11 (trainable)
        ↓
[CLS] Token Extraction → Dropout (0.4)
        ↓
Linear (768 → 192) → GELU → Dropout (0.28)
        ↓
Linear (192 → 2) → Softmax
        ↓
Output: [Ham Probability, Spam Probability]

```



## 🧠 Model Overview

### BERT Model
- **Transformer Backbone**: **DistilBERT-base-uncased** for efficient text understanding  
- **Custom Classifier Head**: Two-layer neural network with dropout regularization  
- **Binary Classification**: Outputs probability scores for ham/spam classification  

### RoBERTa Model (Optimized)
- **Transformer Backbone**: **RoBERTa-base** for robust text understanding  
- **Layer Freezing**: First 6 layers frozen to prevent overfitting
- **Custom Classifier Head**: Simplified two-layer neural network with progressive dropout  
- **Binary Classification**: Outputs calibrated probability scores for ham/spam classification  

## ✨ Key Features

### Common Features
- **Advanced Tokenization**: WordPiece tokenization with attention masks  
- **Cross-Validation**: **5-fold stratified** validation for robust model evaluation  
- **Model Persistence**: Save/load functionality for production deployment  
- **Comprehensive Metrics**: Accuracy, F1-score, precision, recall, and confusion matrices  

### BERT-Specific Features
- **Differential Learning Rates**: Separate rates for BERT layers (1e-5) and classifier (2e-5)  
- **Gradient Clipping**: Prevents exploding gradients with max norm 1.0  
- **Label Smoothing**: Optional cross-entropy enhancement for better calibration  

### RoBERTa-Specific Features (Anti-Overfitting)
- **Smart Model Detection**: Automatically detects existing trained models and skips training  
- **Advanced Anti-Overfitting**: Layer freezing, early stopping, and progressive dropout  
- **Differential Learning Rates**: Separate rates for RoBERTa layers (2e-5) and classifier (5e-5)  
- **Data Augmentation**: Word dropout (10%) and swapping (5%) for training robustness  
- **Label Smoothing**: Cross-entropy enhancement (0.1) for better calibration  
- **Interactive Interface**: Real-time email classification with confidence scores  
- **Batch Processing**: Efficient multi-email prediction with optimized memory usage  

## 🔁 Training Pipeline Overview

### 1. 📂 Data Preparation
- **Text cleaning** and normalization with whitespace handling  
- **Adaptive max length** based on 95th percentile of text lengths (BERT) / optimization based on text distribution (RoBERTa)  
- **Stratified splitting** maintains class balance across folds  
- **Tokenization** with truncation and padding for uniform input size  
- **Smart augmentation** with random word modifications during training (RoBERTa)

### 2. ⚙️ Model Setup
- **Pre-trained weights** with custom classification head  
- **Xavier initialization** for linear layers  
- **Dropout regularization** for optimal performance
  - BERT: (0.3) in both hidden and output layers  
  - RoBERTa: Progressive dropout (0.4 → 0.28) with layer freezing strategy

### 3. 🏋️ Training
- **AdamW optimizer** with weight decay for regularization
  - BERT: Weight decay (0.01)
  - RoBERTa: Optimal weight decay (0.02)
- **Linear warmup scheduler** with 10% warmup steps  
- **Early stopping** based on validation accuracy improvement  
- **Cross-validation** ensures model generalization across data splits  
- **Gradient clipping** (max norm 1.0) for training stability (RoBERTa)

### 4. 🧪 Evaluation
- **Per-fold accuracy** tracking with statistical analysis  
- **Classification reports** with precision/recall per class  
- **Confusion matrix** visualization for error analysis  
- **Test set evaluation** on held-out data for final performance  
- **Training curve** visualization for overfitting detection (RoBERTa)
- **Overfitting tests** with adversarial and out-of-distribution examples (RoBERTa)


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

### 🔷 BERT Model Performance
- **Cross-Validation Accuracy**: **97.56% ± 0.87%** across 5 folds  
- **F1-Score**: **97.51%** weighted average for balanced performance  
- **Test Accuracy**: Consistently exceeds **95%** on held-out data  
- **Inference Speed**: **100-500 emails/second** on GPU, **50-100 emails/second** on CPU  
- **Model Size**: **66MB** compressed checkpoint for efficient deployment  

#### Performance Breakdown:
- **Ham (Legitimate) Precision**: ~98%  
- **Spam Detection Recall**: ~96%  
- **False Positive Rate**: <2%  
- **False Negative Rate**: <4%  

### 🔶 RoBERTa Model Performance (Optimized)
- **Cross-Validation Accuracy**: **99.40% ± 0.20%** across 5 folds  
- **F1-Score**: **99.40%** weighted average for balanced performance  
- **Test Accuracy**: **99.48%** on final held-out data  
- **Training Stability**: Controlled loss convergence without overfitting  
- **Inference Speed**: **200-400 emails/second** on GPU, **50-100 emails/second** on CPU  
- **Model Size**: **500MB** optimized checkpoint for production deployment  

#### Performance Breakdown:
- **Ham (Legitimate) Precision**: ~99.5%  
- **Spam Detection Recall**: ~99.4%  
- **False Positive Rate**: <0.5%  
- **False Negative Rate**: <0.6%  
- **Confidence Calibration**: Appropriate uncertainty on borderline cases

#### Anti-Overfitting Success:
- **Healthy Training Curves**: Gradual loss reduction (45%) vs. original overfitting (99%)
- **Train-Validation Gap**: Minimal difference indicating excellent generalization
- **Stability**: Consistent performance across different data distributions
- **Robustness**: Strong performance on adversarial and out-of-distribution examples
 

## 📄 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

### 💡 Opportunities for Contribution:

- **Model Variants**: Experiment with **ALBERT**, **ELECTRA**, or **DeBERTa** architectures  
- **Advanced Preprocessing**: Implement **HTML tag removal**, **URL normalization**, and **multi-language support**  
- **Ensemble Methods**: Combine BERT and RoBERTa models for improved robustness  
- **Real-time API**: Build **FastAPI** or **Flask** endpoints for production serving  
- **MLOps Integration**: Add **MLflow** tracking, **Docker** containerization, and **Kubernetes** deployment   
- **Interpretability**: Add **SHAP**, **LIME**, or **attention visualization** for model explainability  
- **Model Comparison**: Develop automated benchmarking between BERT and RoBERTa implementations 

### 🔧 How to Contribute:

1. Fork the repository  
2. Create a feature branch  
```bash
git checkout -b feature/new-enhancement
```
3. Implement changes with comprehensive testing and documentation  
4. Submit a pull request with detailed description and performance benchmarks  

**🔷 BERT**: Efficient and lightweight solution for standard email classification tasks
**🔶 RoBERTa**: State-of-the-art performance with advanced anti-overfitting techniques for critical applications

⭐ If this project helps you build better email classification systems, consider giving it a star!



