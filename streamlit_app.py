import streamlit as st
import joblib
import torch
import numpy as np
import re
import nltk
from transformers import BertTokenizer, BertModel
from nltk.corpus import stopwords

st.set_page_config(
    page_title="Twitter Stock Market Sentiment Analyzer",
    layout="centered"
)

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# Load resources
@st.cache_resource
def load_model():
    return joblib.load("rf_pipeline.pkl")

@st.cache_resource
def load_financialbert():
    tokenizer = BertTokenizer.from_pretrained("ahmedrachid/FinancialBERT")
    model = BertModel.from_pretrained("ahmedrachid/FinancialBERT")
    model.eval()
    return tokenizer, model

# Pre-processing the text
@st.cache_resource
def load_stopwords():
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))
    keep_words = {
        "and", "but", "or", "is", "are", "was", "were", "should", "would",
        "could", "a", "an", "the", "in", "on", "at", "with", "for", "of",
        "after", "before", "by", "as"
    }
    return stop_words - keep_words

def smart_clean_text(text, smart_stopwords):
    custom_removals = {
        "via", "eps", "bln", "ipo", "fomc", "mclr", "usmca", "nyse",
        "gapping", "afterhours", "premarket", "today", "week", "month",
        "years", "days", "time", "billion", "million", "trillion", "say"
    }

    try:
        import emoji
        text = emoji.replace_emoji(str(text), replace="")
    except ImportError:
        text = str(text)

    text = re.sub(r"\.\.\.|…|more:", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\$\w+", "", text)
    text = re.sub(r"\b[A-Z]{2,5}:[A-Z]{1,5}\b", "", text)
    text = re.sub(r"\([A-Z]{1,5}\)", "", text)
    text = re.sub(r"\b[A-Z]{2,5}/[A-Z]{2,5}\b", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+(\.\d+)?%?", "", text)
    text = text.lower()
    text = " ".join([w for w in text.split() if w not in smart_stopwords])
    text = " ".join([w for w in text.split() if w not in custom_removals])
    text = " ".join([w for w in text.split() if len(w) > 2])
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Get FinancialBert embedding
def get_embedding(text, tokenizer, bert_model):
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
    mean_pooled = outputs.last_hidden_state.mean(dim=1)
    return mean_pooled.squeeze().numpy()

LABEL_MAP = {
    0: ("Bearish", "red",   "The tweet signals a **negative / downward** market outlook."),
    1: ("Bullish", "green", "The tweet signals a **positive / upward** market outlook."),
    2: ("Neutral", "gray",  "The tweet does not signal a clear market direction."),
}

st.title("Stock Market Sentiment Analysis Using Twitter Data")
st.markdown(
    "Enter a financial tweet or headline and the model will predict whether "
    "the market sentiment is **Bearish**, **Bullish**, or **Neutral**."
)
st.markdown("---")

# Example tweets
examples = [
    "$AAPL — Morgan Stanley raises price target after strong earnings beat.",
    "$TSLA slides after Elon Musk sells another $2B in shares.",
    "Fed holds rates steady at its May meeting.",
]

st.markdown("**Try an example:**")
cols = st.columns(3)
for i, (col, ex) in enumerate(zip(cols, examples)):
    if col.button(f"Example {i+1}", use_container_width=True):
        st.session_state.input_text = ex  

# Text input 
user_input = st.text_area(
    "Enter a financial tweet or headline:",
    value=st.session_state.input_text,
    height=120,
    placeholder="e.g. JPMorgan cuts price target on $BYND after weak quarterly results...",
    key="input_text"
)

predict_btn = st.button("Predict Sentiment", type="primary", use_container_width=True)

# ── Prediction ─────────────────────────────────────────────────────────────────
if predict_btn:
    if not st.session_state.input_text.strip():
        st.warning("Please enter some text before predicting.")
    else:
        with st.spinner("Loading models and generating prediction…"):
            try:
                rf_pipeline   = load_model()
                tokenizer, bert_model = load_financialbert()
                smart_stopwords = load_stopwords()

                # Clean text
                cleaned = smart_clean_text(st.session_state.input_text, smart_stopwords)

                # Generate FinancialBERT embedding
                embedding = get_embedding(cleaned, tokenizer, bert_model)

                # Predict (pipeline handles PCA internally)
                X = embedding.reshape(1, -1)
                pred  = rf_pipeline.predict(X)[0]
                proba = rf_pipeline.predict_proba(X)[0]

                # Display result
                label, color, description = LABEL_MAP[pred]
                st.markdown("---")
                st.markdown(f"### Prediction: :{color}[{label}]")
                st.markdown(description)

                # Confidence bars
                st.markdown("**Confidence scores:**")
                for cls_id, cls_name in [(0, "Bearish"), (1, "Bullish"), (2, "Neutral")]:
                    st.progress(
                        float(proba[cls_id]),
                        text=f"{cls_name}: {proba[cls_id]*100:.1f}%"
                    )

                # Show cleaned text
                with st.expander("🔧 See preprocessed text"):
                    st.code(cleaned if cleaned else "(empty after cleaning)", language=None)

            except Exception as e:
                st.error(f"An error occurred: {e}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Model: Random Forest + PCA trained on FinancialBERT embeddings | "
    "Dataset: Twitter Financial News Sentiment (Hugging Face) | "
    "Classes: 0 = Bearish, 1 = Bullish, 2 = Neutral"
)
