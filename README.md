## Sentiment Analysis Using IMDB Dataset

This project demonstrates sentiment analysis on IMDB movie reviews using machine learning and natural language processing (NLP) techniques. The workflow includes data preprocessing, feature extraction, model training, evaluation, and prediction.

## Table of Contents

- Project Overview
- Dataset
- Requirements
- Workflow
  - 1. Data Loading
  - 2. Preprocessing
  - 3. Feature Extraction
  - 4. Model Training
  - 5. Evaluation
  - 6. Prediction
- Results
- Usage
- References

---

## Project Overview

The goal is to classify IMDB movie reviews as positive or negative using logistic regression. The project leverages Python libraries such as pandas, scikit-learn, NLTK, and spaCy for data handling, NLP, and machine learning.

## Dataset

- **File:** Dataset.csv
- **Columns:**
  - `review`: Text of the movie review
  - `sentiment`: Sentiment label (`positive` or `negative`)

## Requirements

Install the following Python packages:

- pandas
- numpy
- scikit-learn
- nltk
- spacy
- matplotlib
- seaborn

You can install them using:

```bash
pip install pandas numpy scikit-learn nltk spacy matplotlib seaborn
python -m spacy download en_core_web_sm
```

## Workflow

### 1. Data Loading

- Load the dataset from Dataset.csv using pandas.
- Convert sentiment labels to numeric (`positive` → 1, `negative` → 0).

### 2. Preprocessing

- Download NLTK stopwords and punkt tokenizer.
- Remove HTML tags, lowercase text, remove punctuation/numbers.
- Tokenize and remove stopwords.
- Lemmatize tokens using spaCy.

### 3. Feature Extraction

- Use `CountVectorizer` and `TfidfVectorizer` from scikit-learn to convert text into numerical features.

### 4. Model Training

- Split data into training and test sets.
- Train a logistic regression model on the training data.

### 5. Evaluation

- Predict on the test set.
- Calculate Mean Squared Error (MSE), R2 Score, and Accuracy.
- Plot confusion matrix using matplotlib and seaborn.

### 6. Prediction

- Predict sentiment for sample reviews and display results.
- Show predictions for a few test reviews with true and predicted sentiment.

## Results

- The model achieves good accuracy in classifying reviews.
- Confusion matrix and sample predictions are displayed for analysis.

## Usage

1. Place Dataset.csv in the project directory.
2. Open and run main.ipynb in Jupyter Notebook or VS Code.
3. Follow the notebook cells to preprocess data, train the model, and view results.

## References

- [NLTK Documentation](https://www.nltk.org/)
- [spaCy Documentation](https://spacy.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
