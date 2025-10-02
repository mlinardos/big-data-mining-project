# Big Data Mining Techniques Project

A data mining project implementing text classification, locality-sensitive hashing, and time series analysis on a news article dataset using Python and scikit-learn.

## Overview

Three-part project analyzing 159,707 news articles across Business, Entertainment, Health, and Technology categories.

## Part 1: Text Classification

**Goal:** Classify news articles into 4 categories

**Approach:**
- Preprocessing: text normalization, tokenization, lemmatization, stop word removal
- Feature extraction: Binary Bag-of-Words with chi-squared selection (top 5,000 features)
- Models tested: SVM, Random Forest, K-Nearest Neighbors (K=7)

**Results:**
- KNN with Jaccard similarity: **96.3% accuracy** (best model)
- SVM: 94.9% accuracy
- Random Forest: 92.6% accuracy

## Part 2: Locality Sensitive Hashing

**Goal:** Speed up K-NN search using MinHash LSH

**Implementation:**
- MinHash signatures with configurable permutations (16, 32, 64)
- LSH index with threshold values (0.6-0.9)
- Compared against brute-force baseline

**Results** (40% data sample):
- Best configuration: threshold=0.7, permutations=64
- **~100x speedup** (2.71 min vs 274.76 min)
- Trade-off: 22.48% accuracy for 99% time reduction

## Part 3: Dynamic Time Warping

**Goal:** Compute similarity between variable-length time series

**Implementation:**
- Custom DTW algorithm from scratch (no libraries)
- Euclidean distance-based dynamic programming approach
- Processed 1,002 time series pairs in 47 minutes

## Technologies

Python • scikit-learn • DataSketch • NLTK • NumPy • Pandas

## Key Results

- 96.3% classification accuracy on imbalanced dataset
- 100x faster similarity search with LSH
- Custom DTW implementation for time series comparison
