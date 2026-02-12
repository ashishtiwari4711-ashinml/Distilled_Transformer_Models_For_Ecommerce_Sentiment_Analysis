# Sentiment Analysis using Classical ML and Transformer Models

Author: Ashish Tiwari  
Project: NLP Sentiment Analysis  
Frameworks: Scikit-learn, Hugging Face Transformers  

---

## Project Overview

This project implements a comparative sentiment analysis pipeline using:

- Logistic Regression (classical ML baseline)
- DistilBERT (pretrained transformer)
- DistilRoBERTa (pretrained transformer)

The goal was to evaluate how transformer-based models compare against traditional machine learning approaches for text classification.

The project follows a structured NLP pipeline:

Raw Text → Cleaning → Tokenization → Encoding → Model Training → Evaluation → Model Comparison

---

## Problem Statement

Given a dataset of textual reviews, classify each review as positive or negative sentiment.

This is framed as a supervised binary classification task.

---

## Models Implemented

### 1. Logistic Regression (Baseline)

- Text converted using TF-IDF vectorization
- Logistic Regression trained on sparse feature vectors
- Serves as a classical ML benchmark

Purpose:
Establish a strong traditional baseline before applying transformer models.

---

### 2. DistilBERT

- Lightweight version of BERT
- Pretrained on large corpora
- Fine-tuned on the sentiment dataset

Advantages:
- Faster training than full BERT
- Context-aware embeddings
- Captures semantic relationships

---

### 3. DistilRoBERTa

- Distilled version of RoBERTa
- Improved training methodology over BERT
- Fine-tuned on the sentiment dataset

Result:
DistilRoBERTa achieved the highest validation performance among all models.

---

## Why Compare These Models?

The objective was to understand:

- How much performance gain transformers provide over classical ML
- Trade-off between computational cost and accuracy
- Impact of contextual embeddings vs bag-of-words representations

---

## Pipeline Architecture

### Step 1: Data Preprocessing
- Lowercasing
- Cleaning special characters
- Removing noise
- Preparing text for tokenization

### Step 2: Feature Representation

For Logistic Regression:
- TF-IDF vectorization

For Transformer Models:
- Tokenization using pretrained tokenizer
- Attention masks
- Padding and truncation

---

## Training Strategy

Logistic Regression:
- Trained on TF-IDF vectors
- Evaluated using accuracy, precision, recall, F1-score

Transformers:
- Fine-tuned pretrained models
- Used classification head on top of encoder
- Optimized using AdamW
- Evaluated on validation set

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score

These metrics provide balanced evaluation for sentiment classification.

---

## Key Results

- Logistic Regression provided a solid baseline.
- DistilBERT significantly improved contextual understanding.
- DistilRoBERTa achieved the best overall performance.

Conclusion:
Contextual transformer models outperform traditional bag-of-words approaches in sentiment analysis tasks.

---

## Key Learnings

- Pretrained transformers dramatically improve text classification performance.
- Model architecture and pretraining corpus matter.
- Distillation provides strong performance with lower computational cost.
- Baselines are essential for meaningful comparison.


