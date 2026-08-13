# Twitter NLP Financial News Sentiment Analyzer
## Technical Project Documentation

> A production-oriented NLP pipeline for classifying financial tweets and news headlines as **Bearish, Bullish, or Neutral** using FinancialBERT embeddings, PCA dimensionality reduction, and a tuned Random Forest classifier.

---

## 1. Project Overview

### 1.1 Objective

The goal of this project is to automatically classify financial-related text into one of three sentiment categories:

| Label | Sentiment | Interpretation |
|---|---|---|
| `0` | Bearish | Negative or pessimistic financial sentiment |
| `1` | Bullish | Positive or optimistic financial sentiment |
| `2` | Neutral | Informational or sentiment-neutral financial content |

The system accepts a financial tweet or news headline as input and produces:

1. A predicted sentiment class
2. Confidence scores for each sentiment class

The final system combines a domain-specific transformer model for semantic representation with a traditional machine-learning classifier for prediction.

### 1.2 High-Level Architecture

```text
                    ┌─────────────────────┐
                    │ Financial Tweet /   │
                    │ News Headline       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Text Preprocessing  │
                    │                     │
                    │ • URLs              │
                    │ • Emojis            │
                    │ • Mentions          │
                    │ • Hashtags          │
                    │ • Tickers           │
                    │ • Stopwords         │
                    │ • Filler tokens     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    FinancialBERT    │
                    │                     │
                    │ 768-dimensional     │
                    │ contextual vector   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │        PCA          │
                    │                     │
                    │ 768 → optimized     │
                    │ lower-dimensional   │
                    │ representation      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Random Forest     │
                    │                     │
                    │ Tuned using         │
                    │ RandomizedSearchCV  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Sentiment           │
                    │ Prediction           │
                    │                     │
                    │ Bearish / Bullish / │
                    │ Neutral             │
                    └─────────────────────┘
```

---

# 2. Dataset

## 2.1 Source

The project uses the **Twitter Financial News Sentiment** dataset from Hugging Face.

The dataset contains financial tweets and news headlines that have been pre-annotated with sentiment labels.

Original dataset sizes:

- Training set: **9,938 observations**
- Validation/test set: **2,486 observations**

The original labels are:

```text
0 → Bearish
1 → Bullish
2 → Neutral
```

The project ultimately used:

- **5,365 balanced training observations**
- **2,388 testing observations**

The reduction in training observations was primarily caused by class balancing and preprocessing.

---

## 2.2 Class Distribution

The original training data exhibited substantial class imbalance:

| Sentiment | Label | Count |
|---|---:|---:|
| Bearish | 0 | 1,442 |
| Bullish | 1 | 1,923 |
| Neutral | 2 | 6,178 |

The Neutral class represented the majority of the training data.

This imbalance creates a significant modeling problem because a classifier can achieve deceptively strong accuracy by disproportionately predicting the majority class.

### Class Balancing Strategy

Rather than generating synthetic observations using SMOTE, the project randomly sampled **2,000 Neutral observations**.

The resulting training distribution was:

```text
Bearish  → 1,442
Bullish  → 1,923
Neutral  → 2,000
```

SMOTE was also tested but did not produce meaningful performance improvements, so it was not included in the final pipeline.

---

# 3. Exploratory Data Analysis

Before modeling, the dataset was inspected to understand:

- Dataset structure
- Available variables
- Sentiment distribution
- Class imbalance
- Text characteristics
- Potential linguistic features

The primary predictive fields were:

```text
text
sentiment label
```

The EDA confirmed that the primary challenge was not missing structured predictors, but rather extracting meaningful information from unstructured financial language.

The severe class imbalance directly influenced the subsequent modeling strategy.

---

# 4. Text Preprocessing

Raw financial tweets contain substantial noise that can interfere with downstream representation learning.

The preprocessing pipeline standardized the input before embedding generation.

## 4.1 Cleaning Pipeline

The preprocessing pipeline includes:

1. URL removal
2. Emoji removal
3. Mention removal
4. Hashtag handling
5. Stock ticker handling
6. Punctuation/noise removal
7. Lowercasing
8. Whitespace normalization
9. Stopword filtering
10. Financial-domain filler-token removal
11. Filtering of extremely short tokens

Examples of domain-specific filler tokens include terms such as:

```text
eps
ipo
bln
fomc
```

The objective was to reduce irrelevant lexical noise while preserving financially meaningful language.

---

# 5. Feature Engineering

The project explored multiple feature representations before settling on contextual embeddings.

## 5.1 Keyword-Based Features

A simple feature was created to identify the presence of financial sentiment keywords.

Examples included:

```text
buy
sell
gain
loss
upgrade
downgrade
```

A binary `has_keyword` variable was used during the logistic regression analysis.

This approach was useful as a baseline because the resulting model was highly interpretable, but its predictive power was limited.

---

## 5.2 Part-of-Speech Features

Part-of-speech counts were also extracted, including:

- Number of nouns
- Number of adjectives
- Number of verbs

These features were primarily used during the regression analysis to investigate relationships between linguistic structure and tweet length.

The regression model achieved:

```text
Training R² = 0.869
Testing R²  = 0.872
```

Ridge regression with cross-validated regularization produced nearly identical results, suggesting that regularization was not necessary for this low-dimensional feature representation.

These features were not used in the final sentiment classifier because grammatical counts do not adequately represent semantic sentiment.

---

# 6. FinancialBERT Embeddings

## 6.1 Motivation

Traditional representations such as Bag-of-Words were not selected for the final system because they primarily encode word occurrence and do not adequately represent contextual meaning.

Instead, the project uses **FinancialBERT**, a BERT-based model pretrained on financial text.

This provides a representation that is better suited to:

- Financial terminology
- Contextual meaning
- Market-related language
- Sentiment-bearing expressions
- Domain-specific semantics

The preprocessing pipeline feeds cleaned text into FinancialBERT, which generates a dense numerical representation.

---

## 6.2 Embedding Generation

Each input text is tokenized and passed through the transformer model.

The resulting representation is a **768-dimensional vector**.

Conceptually:

```text
Text
  ↓
Tokenizer
  ↓
Token IDs
  ↓
FinancialBERT
  ↓
768-dimensional embedding
```

These embeddings serve as the primary features for downstream machine-learning models.

Because generating embeddings is computationally expensive, the project saves generated embeddings to CSV files so that model development does not require repeatedly executing the transformer.

Example intermediate artifacts:

```text
training_embed.csv
valid_embed.csv
balanced_training_embed.csv
```

This separation also makes the workflow modular:

```text
Preprocessing
      ↓
Embedding Generation
      ↓
Feature Storage
      ↓
Model Training
```

---

# 7. Dimensionality Reduction with PCA

## 7.1 Motivation

FinancialBERT produces 768-dimensional feature vectors.

Training tree-based models directly on the full embedding space resulted in poor generalization.

PCA was therefore introduced to:

- Reduce feature dimensionality
- Remove redundant information
- Reduce noise
- Improve computational efficiency
- Reduce overfitting

The exploratory PCA analysis reduced the embeddings to 100 components, which retained approximately **84.5% of the total variance**.

The scree plot showed diminishing returns after the leading components, supporting the use of dimensionality reduction.

For the final Random Forest pipeline, PCA dimensionality was treated as a tunable parameter rather than permanently fixed at the exploratory value.

The final hyperparameter search selected:

```text
pca__n_components = 150
```

This configuration was incorporated directly into the model-selection pipeline.

---

# 8. Model Development

Several modeling approaches were evaluated.

## 8.1 Logistic Regression

Logistic regression was first used as an interpretable baseline.

A binary version of the task was created:

```text
Neutral     → 1
Not Neutral → 0
```

The model used the `has_keyword` feature.

The coefficient for `has_keyword` was:

```text
-1.0194
```

This suggested that the presence of sentiment-related financial keywords was associated with a lower probability of the Neutral class.

However, predictive performance was limited:

```text
Test Accuracy   = 0.6750
Recall          = 0.9183
Specificity     = 0.2117
AUC             = 0.562
```

The high recall combined with low specificity indicated that the model was strongly influenced by the underlying class distribution.

This experiment demonstrated that simple surface-level features were insufficient for the three-class sentiment task.

---

# 9. Random Forest

## 9.1 Baseline Model

The initial Random Forest operated directly on the 768-dimensional FinancialBERT embeddings.

Initial configuration:

```text
n_estimators = 250
max_depth = 16
class_weight = "balanced"
```

The model achieved strong training performance:

```text
Training Accuracy = 0.87
Training Weighted F1 = 0.88
```

However, cross-validation revealed substantially weaker generalization:

```text
Mean AUC      = 0.541
Mean Accuracy = 0.571
```

This indicated significant overfitting.

The discrepancy between training and validation performance motivated the introduction of PCA and more extensive hyperparameter tuning.

---

# 10. Hyperparameter Optimization

The final Random Forest was optimized using `RandomizedSearchCV` with **10-fold cross-validation**.

The search optimized both Random Forest parameters and PCA dimensionality.

Parameters explored included:

- Number of trees
- Maximum tree depth
- Minimum samples per split
- Minimum samples per leaf
- Number of features considered at each split
- Class weighting
- Cost-complexity pruning
- Number of PCA components

The final configuration was:

```text
n_estimators       = 400
min_samples_split  = 300
min_samples_leaf   = 100
max_features       = "log2"
max_depth          = 10
class_weight       = "balanced"
ccp_alpha          = 0.0001
pca__n_components  = 150
```

The search was designed to optimize generalization rather than maximize training-set performance.

---

# 11. Model Selection

The Random Forest was selected as the final classifier after comparison with:

- Logistic regression
- K-nearest neighbors
- Decision trees
- Neural networks

Random Forest was particularly effective because it can model nonlinear relationships and interactions between PCA-transformed embedding features.

Unlike a single decision tree, the ensemble structure also provides greater robustness.

The project found that KNN struggled with the high-dimensional embedding space, while a single decision tree was less robust than the ensemble approach.

---

# 12. Final Model Performance

The tuned Random Forest achieved the following results on the held-out test data.

## Overall Performance

| Metric | Score |
|---|---:|
| Accuracy | **0.7387** |
| Overall F1 | **0.7385** |
| Macro F1 | **~0.739** |

## Per-Class Performance

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Bearish | 0.69 | 0.74 | 0.71 |
| Bullish | 0.77 | 0.69 | 0.73 |
| Neutral | 0.75 | 0.79 | 0.77 |

## One-vs-Rest AUC

| Class | AUC |
|---|---:|
| Bearish | 0.86 |
| Bullish | 0.84 |
| Neutral | 0.85 |

The consistency across classes is important because accuracy alone can be misleading on an imbalanced dataset.

The model's similar accuracy and macro-F1 values indicate that performance was not dominated by a single sentiment class.

---

# 13. Neural Network Experiment

A feedforward neural network was also evaluated using the PCA-reduced FinancialBERT embeddings.

## Architecture

```text
Input
100 PCA features
      ↓
Dense Layer
128 neurons
ReLU
      ↓
Dense Layer
64 neurons
ReLU
      ↓
Output Layer
3 logits
```

Training configuration:

```text
Loss: Cross-Entropy
Optimizer: SGD
Epochs: 100
Learning rate: 0.01
```

The model achieved:

```text
Accuracy         = 0.6558
Weighted F1      = 0.5194
Weighted Recall  = 0.6558
Weighted Precision = 0.4300
```

The model tended to predict the majority class, demonstrating the difficulty of training neural networks under the dataset's class imbalance.

One limitation of this comparison is that the neural network received substantially less hyperparameter tuning than the Random Forest. Therefore, the experiment does not establish that Random Forest is inherently superior to neural networks for this representation.

---

# 14. Unsupervised Learning

PCA embeddings were also investigated using clustering techniques to better understand the structure of the financial text.

The following methods were evaluated:

- K-Means
- Hierarchical Agglomerative Clustering
- HDBSCAN

---

## 14.1 K-Means

The elbow method suggested:

```text
k = 5
```

while silhouette analysis favored:

```text
k = 2
```

The project selected `k = 5` to investigate more granular structure.

Results:

```text
Inertia     = 889,864.3
Silhouette  = 0.0579
```

The low silhouette score indicated substantial overlap between clusters.

---

## 14.2 Hierarchical Clustering

Ward-linkage hierarchical clustering produced:

```text
Silhouette = 0.0368
```

This provided additional evidence that strong, clearly separated clusters did not exist in the embedding space at this level of granularity.

---

## 14.3 HDBSCAN

HDBSCAN was particularly useful because it can identify ambiguous observations as noise rather than forcing every observation into a cluster.

Results:

```text
Clusters excluding noise = 3
Noise observations       = 2,470
Noise percentage         = 25.88%
Silhouette excluding noise = 0.1812
```

The three non-noise clusters had sizes:

```text
17
7,029
27
```

Although most observations remained ambiguous, HDBSCAN produced the strongest clustering separation among the evaluated methods.

This is consistent with the overlapping nature of financial sentiment, where individual pieces of text may not belong cleanly to discrete semantic groups.

---

# 15. Repository Structure

The repository is organized around reproducible experimentation and application deployment.

```text
Twitter-NLP-Financial-News-Sentiment/
│
├── .devcontainer/
│   └── devcontainer.json
│
├── .github/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── preprocessing.ipynb
│   └── random_forest_model.ipynb
│
├── requirements.txt
├── streamlit_app.py
├── rf_pipeline.pkl
├── LICENSE
└── README.md
```

The repository currently contains separate notebooks for exploratory analysis, preprocessing, and Random Forest development.

### Key Components

#### `notebooks/`

Contains the research and development workflow.

- `EDA.ipynb` — exploratory analysis and dataset investigation
- `preprocessing.ipynb` — text cleaning and FinancialBERT embedding generation
- `random_forest_model.ipynb` — PCA, model training, hyperparameter tuning, and evaluation

#### `streamlit_app.py`

Provides the interactive application layer.

Users can enter a financial tweet or headline and receive a sentiment prediction.

#### `rf_pipeline.pkl`

Serialized trained machine-learning pipeline containing the final Random Forest/PCA model used by the application.

#### `requirements.txt`

Defines the Python dependencies required to reproduce the project environment.

#### `.devcontainer/`

Provides containerized development configuration for a reproducible development environment.

---

# 16. Reproducibility

## 16.1 Environment Setup

Clone the repository:

```bash
git clone https://github.com/SummerKirbs/Twitter-NLP-Financial-News-Sentiment.git
cd Twitter-NLP-Financial-News-Sentiment
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The project uses Python-based tooling including PyTorch, Transformers, scikit-learn, NLTK, Emoji, Joblib, and Streamlit.

---

# 17. Reproducing the Modeling Pipeline

The complete modeling workflow consists of the following stages.

### Stage 1 — Data Loading

Load:

```text
sent_train.csv
sent_valid.csv
```

Each observation contains financial text and its corresponding sentiment label.

### Stage 2 — Preprocessing

Apply the project's text-cleaning pipeline.

```text
Raw text
   ↓
Remove URLs
   ↓
Remove emojis/noise
   ↓
Normalize text
   ↓
Remove selected stopwords
   ↓
Remove domain-specific filler tokens
```

### Stage 3 — Embedding Generation

Run the cleaned text through FinancialBERT.

```text
Clean text
   ↓
FinancialBERT
   ↓
768-dimensional embedding
```

Save the resulting representations for downstream experimentation.

### Stage 4 — Class Balancing

Randomly downsample the Neutral class to 2,000 observations.

### Stage 5 — PCA

Transform the embedding representation into a lower-dimensional feature space.

### Stage 6 — Hyperparameter Search

Run:

```python
RandomizedSearchCV(
    ...,
    cv=10
)
```

using macro-F1 as the primary model-selection criterion.

### Stage 7 — Final Training

Train the optimized Random Forest using the selected PCA and classifier parameters.

### Stage 8 — Evaluation

Evaluate the final model using:

- Accuracy
- Precision
- Recall
- F1
- AUC
- Confusion matrix

---

# 18. Running the Application

The repository includes a Streamlit application for interactive inference.

Start the application with:

```bash
streamlit run streamlit_app.py
```

The application opens a browser interface where users can enter financial text.

Example:

```text
Input:
AAPL raises guidance after strong quarterly earnings.
```

The application returns a predicted sentiment and associated confidence scores.

The current repository README documents the same inference workflow and example inputs.

---

# 19. Design Decisions

## Why FinancialBERT?

Financial language differs significantly from general-purpose language.

Terms such as:

```text
earnings
guidance
downgrade
valuation
EPS
bullish
bearish
```

carry domain-specific meaning.

FinancialBERT provides contextual embeddings specifically suited to financial text rather than relying exclusively on generic language representations.

---

## Why PCA?

Directly training a Random Forest on the original 768-dimensional embeddings resulted in poor validation performance.

PCA reduced the feature space and helped control overfitting.

It also reduced computational complexity while retaining a substantial portion of the embedding variance.

---

## Why Random Forest?

Random Forest provided the strongest combination of:

- Generalization
- Nonlinear modeling capability
- Robustness
- Relatively straightforward tuning
- Performance across all three sentiment classes

The model also performed better than the evaluated logistic regression and neural-network configurations.

---

# 20. Limitations

The final system has several important limitations.

## 20.1 Class Imbalance

Although the training data was balanced through downsampling, the original dataset was heavily dominated by Neutral examples.

This can still affect minority-class representation.

## 20.2 Information Loss from PCA

PCA compresses the original embedding space.

Although this improves generalization, some fine-grained semantic information may be discarded.

## 20.3 Embedding Computational Cost

FinancialBERT inference is computationally expensive compared with traditional text representations.

Embedding generation is therefore separated from downstream model training.

## 20.4 Generalization

The final model performs well on the held-out dataset, but performance on financial language outside the dataset may differ.

Financial sentiment can change rapidly with market context, terminology, and events.

## 20.5 Model Comparison Constraints

The Random Forest received substantially more hyperparameter optimization than the neural-network experiment.

Therefore, the results should not be interpreted as proving that Random Forest is universally better than neural networks for financial sentiment classification.

These limitations are consistent with the project's original evaluation of class imbalance, high-dimensional embeddings, computational cost, and generalization risk.

---

# 21. Future Improvements

Potential improvements include:

### Better class-balancing strategies

Evaluate:

- Class-weighted objectives
- Balanced sampling
- Focal loss
- More sophisticated oversampling techniques

### End-to-end transformer fine-tuning

Rather than using FinancialBERT strictly as a feature extractor, fine-tune the transformer directly on the project's three-class classification objective.

### Improved neural-network baseline

Conduct a more comprehensive search over:

- Architecture
- Optimizer
- Learning rate
- Batch size
- Dropout
- Weight decay
- Class-weighted loss

### More robust evaluation

Use additional metrics such as:

- Macro-F1
- Per-class precision/recall
- Calibration
- Confusion matrices
- Precision-recall curves

### Temporal validation

Financial sentiment is inherently time-dependent.

A future implementation should consider chronological train/test splits to determine whether the model generalizes to future market conditions rather than only randomly held-out observations.

---

# 22. Reproducible Development Workflow

The recommended workflow for modifying the project is:

```text
             ┌───────────────┐
             │    Dataset    │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │      EDA      │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │ Preprocessing │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │ FinancialBERT │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │      PCA      │
             └───────┬───────┘
                     │
                     ▼
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────┐        ┌────────────────┐
│ Model         │        │ Unsupervised   │
│ Experiments   │        │ Analysis       │
└───────┬───────┘        └────────────────┘
        │
        ▼
┌───────────────────┐
│ Hyperparameter    │
│ Optimization      │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Final Random      │
│ Forest Pipeline   │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Streamlit         │
│ Application       │
└───────────────────┘
```

This architecture separates **research**, **feature generation**, **model development**, and **inference**, making the repository easier to maintain and extend.

---

# 23. Technical Summary

| Component | Implementation |
|---|---|
| Problem | 3-class financial sentiment classification |
| Input | Financial tweets / news headlines |
| Classes | Bearish, Bullish, Neutral |
| Dataset | Twitter Financial News Sentiment |
| Text Representation | FinancialBERT |
| Raw Embedding Size | 768 dimensions |
| Dimensionality Reduction | PCA |
| Final PCA Size | 150 components |
| Class Balancing | Neutral downsampling |
| Primary Model | Random Forest |
| Optimization | RandomizedSearchCV |
| Cross-Validation | 10-fold |
| Selection Metric | Macro-F1 |
| Final Accuracy | 0.7387 |
| Final F1 | 0.7385 |
| AUC | 0.84–0.86 |
| Deployment | Streamlit |
| Model Serialization | Joblib / `.pkl` |

---

# 24. Project Takeaways

This project demonstrated a complete machine-learning workflow for unstructured financial text:

1. **EDA identified severe class imbalance.**
2. **Text preprocessing reduced domain-specific noise.**
3. **FinancialBERT converted unstructured text into contextual numerical representations.**
4. **PCA reduced the dimensionality and helped control overfitting.**
5. **Multiple supervised models were evaluated.**
6. **Random Forest provided the strongest overall performance.**
7. **Randomized hyperparameter search improved generalization.**
8. **Clustering provided additional insight into the overlapping structure of financial sentiment.**
9. **A Streamlit application converted the trained pipeline into an interactive inference system.**

The resulting system achieved approximately **74% accuracy and 74% F1** while maintaining relatively consistent performance across Bearish, Bullish, and Neutral classes.

The primary engineering lesson is that model performance depended not only on the choice of classifier, but on the interaction between **domain-specific representation learning, dimensionality reduction, class balancing, and hyperparameter optimization**.