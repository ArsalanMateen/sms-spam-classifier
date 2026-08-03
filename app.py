import streamlit as st
import joblib
import string
import os
import re
import nltk
import warnings
import html
from nltk.corpus import stopwords

warnings.filterwarnings('ignore', category=UserWarning)

@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass

download_nltk_resources()

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()
_stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    # null and type safety check
    if not text or not isinstance(text, str):
        return ""
    
    # decode html entities and lowercase
    text = html.unescape(text).lower()

    # tokenize
    tokens = nltk.word_tokenize(text)

    # filter alphanumeric tokens
    tokens = [word for word in tokens if word.isalnum()]

    # remove stopwords and punctuation via list comprehension
    tokens = [word for word in tokens if word not in _stop_words and word not in string.punctuation]

    # apply porter stemming
    tokens = [ps.stem(word) for word in tokens]

    return ' '.join(tokens)


def extract_signals(raw_text):
    digits = sum(1 for c in raw_text if c.isdigit())
    currency = len(re.findall(r'[\$£€¥₹]', raw_text))
    caps_words = len([w for w in raw_text.split() if w.isupper() and len(w) > 1])
    urls = len(re.findall(r'https?://\S+|www\.\S+', raw_text, re.IGNORECASE))
    char_count = len(raw_text)
    word_count = len(raw_text.split())
    return {
        'Digits': digits,
        'Currency symbols': currency,
        'Capitalized words': caps_words,
        'URLs': urls,
        'Characters': char_count,
        'Words': word_count,
    }


MODEL_REGISTRY = {'Linear SVC': 'linear_svc_model.pkl'}

@st.cache_resource
def load_assets():
    tfidf_path = 'tfidf_vectorizer.pkl'
    if not os.path.exists(tfidf_path):
        return None, {}

    try:
        tfidf = joblib.load(tfidf_path)
    except Exception:
        return None, {}

    models = {}
    for label, path in MODEL_REGISTRY.items():
        if os.path.exists(path):
            try:
                models[label] = joblib.load(path)
            except Exception:
                pass
    return tfidf, models

st.set_page_config(
    page_title="SMS Spam Classifier",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] { display: none !important; }

    header[data-testid="stHeader"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    #mainmenu { display: none !important; visibility: hidden !important; }

    .block-container {
        max-width: 1040px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    html, body, [class*="css"] {
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI",
                     Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 1.25rem;
        margin-bottom: 1.75rem;
    }
    .app-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.025em;
        margin: 0;
        line-height: 1.2;
    }
    .header-links { display: flex; gap: 0.6rem; }
    .header-links a {
        color: #475569;
        transition: color 0.15s;
        display: flex;
        align-items: center;
    }
    .header-links a:hover { color: #e2e8f0; }

    .stat-row {
        display: flex;
        gap: 0.6rem;
        margin-bottom: 1.5rem;
    }
    .stat-card {
        flex: 1;
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
    }
    .stat-label {
        font-size: 0.65rem;
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }
    .stat-value {
        font-size: 0.95rem;
        font-weight: 600;
        color: #cbd5e1;
        margin-top: 0.15rem;
    }

    .sec-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.6rem;
    }

    .verdict-spam {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.35);
        color: #fca5a5;
        font-size: 0.85rem;
        font-weight: 700;
        padding: 0.55rem 1.1rem;
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        letter-spacing: 0.04em;
    }
    .verdict-ham {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #6ee7b7;
        font-size: 0.85rem;
        font-weight: 700;
        padding: 0.55rem 1.1rem;
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        letter-spacing: 0.04em;
    }

    .verdict-detail {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-top: 0.5rem;
        line-height: 1.55;
    }

    .conf-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 0.4rem;
    }
    .conf-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748b;
        min-width: 70px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .conf-track {
        flex: 1;
        height: 6px;
        background: #1e293b;
        border-radius: 3px;
        overflow: hidden;
    }
    .conf-fill-spam {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #dc2626, #f87171);
        transition: width 0.4s ease;
    }
    .conf-fill-ham {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #059669, #34d399);
        transition: width 0.4s ease;
    }
    .conf-pct {
        font-size: 0.78rem;
        font-weight: 600;
        color: #cbd5e1;
        min-width: 42px;
        text-align: right;
    }

    .signal-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.3rem 1.2rem;
    }
    .signal-item {
        display: flex;
        justify-content: space-between;
        padding: 0.35rem 0;
        border-bottom: 1px solid #1e293b;
        font-size: 0.78rem;
    }
    .signal-name { color: #94a3b8; }
    .signal-val  { color: #e2e8f0; font-weight: 600; font-variant-numeric: tabular-nums; }
    .signal-val.hot { color: #fbbf24; }

    .sample-btn-wrap div.stButton > button {
        text-align: left !important;
        font-size: 0.78rem;
        padding: 0.5rem 0.75rem;
        border: 1px solid #1e293b;
        background: #0f172a;
        color: #94a3b8;
        border-radius: 8px;
        transition: all 0.15s;
    }
    .sample-btn-wrap div.stButton > button:hover {
        border-color: #334155;
        background: #1e293b;
        color: #e2e8f0;
    }

    div.stButton > button[kind="primary"] {
        border-radius: 8px;
        font-weight: 600;
    }
    div.stButton > button {
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid #1e293b;
        background: #0f172a;
        color: #94a3b8;
    }
    div.stButton > button:hover {
        border-color: #334155;
        background: #1e293b;
        color: #e2e8f0;
    }

    @media (max-width: 768px) {
        .block-container { padding-top: 1rem; padding-left: 0.75rem; padding-right: 0.75rem; }
        .stat-row { flex-wrap: wrap; }
        .stat-card { min-width: 45%; }
        .signal-grid { grid-template-columns: 1fr; }
    }
</style>
""", unsafe_allow_html=True)

tfidf, models = load_assets()
model_names = list(models.keys()) if models else []

st.markdown("""
<div class="app-header">
    <div>
        <div class="app-title">SMS Spam Classifier</div>
    </div>
    <div class="header-links">
        <a href="https://archive.ics.uci.edu/dataset/228/sms+spam+collection" target="_blank" title="UCI Dataset">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <ellipse cx="12" cy="5" rx="9" ry="3"></ellipse>
                <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path>
                <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path>
            </svg>
        </a>
        <a href="https://github.com/ArsalanMateen/sms-spam-classifier" target="_blank" title="GitHub Repository">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
            </svg>
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

METRICS = {
    'Linear SVC': {'accuracy': '98.4%', 'precision': '98.3%', 'recall': '88.3%', 'f1': '0.930'},
}

selected_model_name = "Linear SVC"

active_model = models.get(selected_model_name)
m = METRICS.get(selected_model_name, {})

st.markdown(f"""
<div class="stat-row">
    <div class="stat-card">
        <div class="stat-label">Model</div>
        <div class="stat-value">{selected_model_name}</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Feature Extractor</div>
        <div class="stat-value">TF-IDF</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Accuracy</div>
        <div class="stat-value">{m.get('accuracy', '—')}</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Precision</div>
        <div class="stat-value">{m.get('precision', '—')}</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Recall</div>
        <div class="stat-value">{m.get('recall', '—')}</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">F1-Score</div>
        <div class="stat-value">{m.get('f1', '—')}</div>
    </div>
</div>
""", unsafe_allow_html=True)

if "input_text_val" not in st.session_state:
    st.session_state["input_text_val"] = ""

def load_preset(sample_text):
    st.session_state["input_text_val"] = sample_text


left_col, right_col = st.columns([1.9, 1], gap="medium")

with right_col:
    st.markdown('<div class="sec-title">Try a sample</div>', unsafe_allow_html=True)

    samples = [
        ("Account alert scam", "Your Wells Fargo account has been temporarily locked due to unusual activity. Verify your identity now at http://wellsfargo-secure-auth.com/verify to restore access."),
        ("Prize scam", "As a valued customer you have been selected to receive a £900 prize reward! To claim visit http://bit.ly/3xYz9a2 and enter your details."),
        ("Casual conversation", "Hey, are we still meeting up for dinner tonight around 7 PM?"),
        ("Quick update", "Running about 15 mins late. Traffic is terrible today. Be there soon!"),
    ]

    st.markdown('<div class="sample-btn-wrap">', unsafe_allow_html=True)
    for label, text in samples:
        st.button(label, on_click=load_preset, args=(text,), use_container_width=True, key=f"sample_{label}")
    st.markdown('</div>', unsafe_allow_html=True)

with left_col:
    st.markdown('<div class="sec-title">Message input</div>', unsafe_allow_html=True)
    user_input = st.text_area(
        "Message text",
        value=st.session_state["input_text_val"],
        height=180,
        placeholder="Paste or type the SMS message to analyse",
        label_visibility="collapsed",
    )

    btn1, btn2, _ = st.columns([1, 1, 2])
    with btn1:
        predict_btn = st.button("Analyse", type="primary", use_container_width=True)
    with btn2:
        st.button("Clear", on_click=load_preset, args=("",), use_container_width=True)

    if predict_btn:
        if not user_input.strip():
            st.warning("Please enter a message to analyse.")
        elif active_model is None or tfidf is None:
            st.error("Model assets could not be loaded. Ensure `.pkl` files are present.")
        else:
            with st.spinner("Analysing message…"):
                transformed = preprocess_text(user_input)
                vector_input = tfidf.transform([transformed])
                prediction = active_model.predict(vector_input)[0]
                # linearsvc doesn't have predict_proba by default.
                if hasattr(active_model, "predict_proba"):
                    probabilities = active_model.predict_proba(vector_input)[0]
                    spam_pct = float(probabilities[1]) * 100
                    ham_pct  = float(probabilities[0]) * 100
                else:
                    # fallback to decision_function and sigmoid
                    import math
                    decision = active_model.decision_function(vector_input)[0]
                    # sigmoid
                    prob_spam = 1 / (1 + math.exp(-decision))
                    spam_pct = prob_spam * 100
                    ham_pct = (1 - prob_spam) * 100

            st.markdown("<hr style='border-color:#1e293b; margin:1.2rem 0;'>", unsafe_allow_html=True)

            # verdict
            if prediction == 1:
                st.markdown(
                    '<div class="verdict-spam">SPAM</div>'
                    '<div class="verdict-detail">This message exhibits patterns consistent with spam or fraud, such as unsolicited offers, embedded links, or urgency cues.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="verdict-ham">NORMAL</div>'
                    '<div class="verdict-detail">This message appears to be a normal, personal communication with no strong spam indicators.</div>',
                    unsafe_allow_html=True,
                )

            # confidence bars
            st.markdown(f"""
            <div style="margin-top:0.9rem;">
                <div class="conf-row">
                    <span class="conf-label">Spam</span>
                    <div class="conf-track"><div class="conf-fill-spam" style="width:{spam_pct:.1f}%"></div></div>
                    <span class="conf-pct">{spam_pct:.1f}%</span>
                </div>
                <div class="conf-row" style="margin-top:0.35rem;">
                    <span class="conf-label">Legit</span>
                    <div class="conf-track"><div class="conf-fill-ham" style="width:{ham_pct:.1f}%"></div></div>
                    <span class="conf-pct">{ham_pct:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # signal breakdown
            signals = extract_signals(user_input)
            st.markdown("<hr style='border-color:#1e293b; margin:1.1rem 0 0.7rem;'>", unsafe_allow_html=True)
            st.markdown('<div class="sec-title">Signal breakdown</div>', unsafe_allow_html=True)

            items_html = ""
            for name, val in signals.items():
                hot_class = ' hot' if val > 0 and name in ('Digits', 'Currency symbols', 'Capitalized words', 'URLs / links') else ''
                items_html += f'<div class="signal-item"><span class="signal-name">{name}</span><span class="signal-val{hot_class}">{val}</span></div>\n'

            st.markdown(f'<div class="signal-grid">{items_html}</div>', unsafe_allow_html=True)
