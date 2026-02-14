# CS 288 - Natural Language Processing (UC Berkeley, Spring 2026)

## Assignment 1: Text Classification & Language Modeling

Built text classification and language modeling systems from scratch using both classical ML and neural approaches.

### Part 1: N-Gram Language Models
- **N-Gram Model**: Implemented character-level n-gram language model with add-alpha smoothing for text generation
- **Neural N-Gram Model**: Feedforward neural language model with embedding layer, weight tying, and dropout regularization trained on WikiText-2

### Part 2: Text Classification
- **Perceptron Classifier**: Averaged perceptron with custom feature engineering (bag-of-words, bigrams, negation detection, punctuation features, word suffixes) for SST-2 sentiment analysis and 20 Newsgroups topic classification
- **Multi-Layer Perceptron**: PyTorch MLP with word embeddings, max+mean pooling, BatchNorm, dropout, AdamW optimization, learning rate scheduling, and early stopping

### Datasets
- **SST-2**: Binary sentiment classification (positive/negative movie reviews)
- **20 Newsgroups**: Multi-class topic classification (20 categories)

### Tech Stack
Python, PyTorch, NumPy, pandas
