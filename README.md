Test out the model here! [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://document-question-answering-template.streamlit.app/)


## Overview
The project classifies **financial sentiment** (Bearish, Bullish, Neutral) from Twitter posts and news headlines using **FinancialBERT embeddings** and various ML models.

---

## Dataset
This project uses the **Twitter Financial News Sentiment dataset** from HuggingFace:
[Dataset Link](http://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)  
An overview of the dataset and its labels:
  - Training: 9,938 samples
  - Testing: 2,486 samples
  - Labels: Bearish (0), Bullish (1), Neutral (2)

The neutral class was randomly downsampled to 2,000 to address severe class imbalance.

---

## Methodology
1. Text Preprocessing:  
    - Removed URLs, emojis, stock tickers ($AAPL), hashtags, and mentions
    - Removed finance-specific filler tokens (e.g., eps, ipo, bln, fomc)
    - Lowercased, normalized whitespace, filtered short tokens and stopwords (with key financial connectors preserved)
2. Embeddings:  
    - FinancialBERT (ahmedrachid/FinancialBERT) was chosen for its financial domain pretraining
3. Dimensionality Reduction:  
    - PCA reduced 768 → 150 components
    - Retained the most meaningful variance while minimizing noise-induced overfitting
4. Models Evaluated:  
    - Logistic Regression using keywords
    - *Random Forest (best performing)**
    - Neural Networks
---

## Results
Tuned via RandomizedSearchCV with 10-Fold Cross Validation scored on macro-F1
  - Accuracy: **0.739**
  - Macro F1: **0.739**
  - AUC (per class): **0.84-0.86**

The consistently high AUC values across all classes indicate the model learned meaningful patterns rather than overfitting, and can reliably distinguish each class across a wide range of decision thresholds.

---

## Files
  - `CSM148_Project_Report.pdf`: Full report
  - `notebooks/` - EDA, Models, & Analysis
---

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
