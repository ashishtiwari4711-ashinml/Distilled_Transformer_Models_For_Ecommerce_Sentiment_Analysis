# Sentiment Analysis using NLP and Machine Learning

Author: Ashish Tiwari  
Project: Sentiment Analysis (Natural Language Processing) 

---

## Project Overview

This project implements a complete **Natural Language Processing (NLP) sentiment analysis pipeline** to classify text reviews into sentiment categories.

The goal is to transform raw textual data into meaningful numerical representations and apply classical machine learning algorithms to predict sentiment.

The project demonstrates the full ML lifecycle:

Raw Text → Cleaning → Tokenization → Vectorization → Model Training → Evaluation → Comparison

---

## Problem Statement

Given a dataset of textual reviews, predict whether each review expresses positive or negative sentiment.

This is framed as a **supervised text classification problem**.

---

## Key Objectives

- Perform extensive text preprocessing
- Convert text into numerical features using TF-IDF
- Train multiple ML classifiers
- Compare model performance using accuracy and classification reports
- Identify the best-performing model

---

## Pipeline Architecture

1. Load dataset  
2. Clean text (lowercasing, punctuation removal, stopwords removal)  
3. Tokenize sentences  
4. Convert text to TF-IDF vectors  
5. Split into training and testing sets  
6. Train ML models  
7. Evaluate using accuracy and classification metrics  


---

## Technologies Used

- Python  
- Pandas / NumPy  
- NLTK  
- Scikit-learn  
- Matplotlib / Seaborn  

---

## Text Preprocessing

The following steps are applied:

### Lowercasing
Ensures consistency across tokens.

### Removing punctuation and special characters
Reduces noise.

### Stopword removal
Removes common words (e.g., "the", "is") that carry little sentiment.

### Tokenization
Splits sentences into individual words.

These steps significantly improve feature quality.

---

## Feature Engineering (TF-IDF)

TF-IDF (Term Frequency–Inverse Document Frequency) converts text into numerical vectors.

Why TF-IDF?

- Penalizes common words
- Highlights important terms
- Works well with classical ML models
- Lightweight compared to embeddings

---

## Models Trained

The following classifiers were implemented:

- Logistic Regression  
- Naive Bayes  
- Support Vector Machine  

Each model was trained on TF-IDF features and evaluated on unseen test data.

---

## Evaluation Metrics

Models are compared using:

### Accuracy
Overall correctness.

### Precision
How many predicted positives were correct.

### Recall
How many actual positives were detected.

### F1-score
Balance between precision and recall.

A full classification report is generated for each model.

---

## Results Summary

Support Vector Machine achieved the strongest overall performance.

Key observation:

Proper text preprocessing and TF-IDF feature extraction had a larger impact on performance than model choice.

---



