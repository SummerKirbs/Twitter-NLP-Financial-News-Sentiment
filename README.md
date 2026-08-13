# Twitter NLP Financial News Sentiment Analyzer 

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://twitter-nlp.streamlit.app/)

This project classifies **financial sentiment** (Bearish, Bullish, Neutral) from Twitter posts and news headlines using **FinancialBERT embeddings** and a **Random Forest** machine learning model.

## Features

*   **Financial Sentiment Analysis**: Accurately predicts market sentiment as Bearish, Bullish, or Neutral from financial news and tweets.
*   **FinancialBERT Embeddings**: Leverages the power of a domain-specific pre-trained model for rich feature extraction from financial text.
*   **Random Forest Model**: Employs a robust Random Forest classifier, identified as the best performing model during evaluation.
*   **PCA for Dimensionality Reduction**: Utilizes Principal Component Analysis to reduce feature dimensionality, enhancing model efficiency and potentially mitigating overfitting.
*   **Interactive Streamlit Application**: Provides a user-friendly web interface for real-time sentiment analysis.
*   **Extensive Text Preprocessing**: Implements a sophisticated text cleaning pipeline to handle URLs, mentions, hashtags, stock tickers, emojis, and domain-specific jargon.

## Tech Stack

*   **Programming Language**: Python
*   **Frameworks/Libraries**: Streamlit, PyTorch, scikit-learn, NLTK, Transformers, Joblib, Emoji
*   **Model**: FinancialBERT, Random Forest, PCA
*   **Dataset**: Hugging Face Twitter Financial News Sentiment

## Table of Contents

*   [Overview](#overview)
*   [Dataset](#dataset)
*   [Methodology](#methodology)
*   [Results](#results)
*   [Project Structure](#project-structure)
*   [Installation](#installation)
*   [Usage](#usage)
*   [Contributing](#contributing)
*   [License](#license)
*   [Important Links](#important-links)
*   [Footer](#footer)

## Overview

The project is designed to analyze financial sentiment from text data, specifically focusing on tweets and news headlines. It aims to provide a reliable sentiment score (Bearish, Bullish, or Neutral) crucial for understanding market trends and making informed decisions.

## Dataset

This project utilizes the **Twitter Financial News Sentiment dataset** available on HuggingFace:

[Dataset Link](http://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)

The dataset comprises:

*   **Training Samples**: 9,938
*   **Testing Samples**: 2,486
*   **Labels**: Bearish (0), Bullish (1), Neutral (2)

To address a severe class imbalance, the neutral class was randomly downsampled to 2,000 samples during training.

## Methodology

1.  **Text Preprocessing**: A thorough cleaning process is applied to the input text, including:
    *   Removal of URLs, emojis, stock tickers (e.g., `$AAPL`), hashtags, and mentions.
    *   Exclusion of finance-specific filler tokens (e.g., `eps`, `ipo`, `bln`, `fomc`).
    *   Lowercasing, normalization of whitespace, and filtering of short tokens and common stopwords (while preserving key financial connectors).
2.  **Embeddings**: **FinancialBERT (ahmedrachid/FinancialBERT)** is used to generate contextualized embeddings, capturing the nuances of financial language.
3.  **Dimensionality Reduction**: **Principal Component Analysis (PCA)** is employed to reduce the dimensionality of the embeddings from 768 to 150 components. This step helps retain the most significant variance and minimize noise.
4.  **Models Evaluated**: Several machine learning models were assessed, with the **Random Forest** demonstrating the best performance.
    *   Logistic Regression (using keywords)
    *   *Random Forest (best performing)*
    *   Neural Networks

## Results

The model was tuned using RandomizedSearchCV with 10-Fold Cross-Validation, scored on macro-F1:

*   **Accuracy**: 0.739
*   **Macro F1**: 0.739
*   **AUC (per class)**: 0.84-0.86

The consistently high AUC values across all classes suggest that the model has learned meaningful patterns and can reliably distinguish between different sentiment classes across a wide range of decision thresholds.

## Project Structure

```plaintext
. 
├── .devcontainer/
│   └── devcontainer.json
├── notebooks/
│   ├── EDA.ipynb
│   ├── preprocessing.ipynb
│   └── random_forest_model.ipynb
├── requirements.txt
├── streamlit_app.py
└── rf_pipeline.pkl
└── LICENSE
```

*   `requirements.txt`: Lists all project dependencies.
*   `streamlit_app.py`: The main application file for the Streamlit interface.
*   `notebooks/`: Contains Jupyter notebooks for Exploratory Data Analysis (EDA), preprocessing, and model development.
*   `rf_pipeline.pkl`: Serialized file containing the trained Random Forest model pipeline (including PCA).
*   `.devcontainer/`: Configuration for developing inside a container.
*   `LICENSE`: Project license information.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/SummerKirbs/Twitter-NLP-Financial-News-Sentiment.git
    cd Twitter-NLP-Financial-News-Sentiment
    ```

2.  **Install dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Usage

### Running the Streamlit Application

Once the dependencies are installed, you can run the Streamlit application from your terminal:

```bash
streamlit run streamlit_app.py
```

This command will launch the application in your web browser, allowing you to input financial news headlines or tweets and get sentiment predictions.

### How to use it:

1.  Navigate to the `Streamlit App` page after running the command above.
2.  Enter a financial tweet or headline into the provided text area.
3.  Click the `Predict Sentiment` button.
4.  The application will display the predicted sentiment (Bearish, Bullish, or Neutral) along with confidence scores.

**Example Usage:**

*   **Input**: `$AAPL — Morgan Stanley raises price target after strong earnings beat.`
    **Output**: Bullish

*   **Input**: `$TSLA slides after Elon Musk sells another $2B in shares.`
    **Output**: Bearish

*   **Input**: `Fed holds rates steady at its May meeting.`
    **Output**: Neutral

## 👩‍💻 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any suggestions, improvements, or bug reports.

## License

This project is licensed under the **Apache License 2.0**. See the `LICENSE` file for more details.

## Important Links

*   **Live Demo**: [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://twitter-nlp.streamlit.app/)
*   **Dataset**: [Hugging Face Twitter Financial News Sentiment](http://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)

## Footer

--- 

**Twitter NLP Financial News Sentiment Analyzer**

*   **Repository**: [SummerKirbs/Twitter-NLP-Financial-News-Sentiment](https://github.com/SummerKirbs/Twitter-NLP-Financial-News-Sentiment)
*   **Author**: [SummerKirbs](https://github.com/SummerKirbs)

---
**<p align="center">Generated by [ReadmeCodeGen](https://www.readmecodegen.com/)</p>**
