import streamlit as st
import requests
import pandas as pd

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Reddit Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for clean sentiment badges
st.markdown("""
<style>
    .sentiment-badge {
        font-size: 1.4rem;
        font-weight: 700;
        padding: 0.6rem 1.4rem;
        border-radius: 8px;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .badge-positive {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .badge-negative {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .badge-neutral {
        background-color: #e2e3e5;
        color: #383d41;
        border: 1px solid #d6d8db;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar: Backend Connection & Server Status
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://redditinc.com/assets/images/site/reddit-logo.png", width=140)
    st.title("Settings")
    
    api_url = st.text_input(
        "FastAPI Backend URL",
        value="http://127.0.0.1:8000",
        help="URL of your running FastAPI sentiment analysis server",
    )

    st.divider()
    st.subheader("Server Status")
    
    server_online = False
    try:
        res = requests.get(f"{api_url.rstrip('/')}/info", timeout=2)
        if res.status_code == 200:
            data = res.json()
            st.success("🟢 API Server is Online")
            st.caption(f"• Model Loaded: `{data.get('model_loaded')}`")
            st.caption(f"• Vectorizer Loaded: `{data.get('vectorizer_loaded')}`")
            server_online = True
        else:
            st.warning(f"🟡 API status: {res.status_code}")
    except Exception:
        st.error("🔴 API Server is Offline")
        st.caption("Start the backend with:")
        st.code("python api.py", language="bash")

    st.divider()
    st.caption("Reddit Comment Sentiment Classifier (TF-IDF + LightGBM)")

# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.title("💬 Reddit Comment Sentiment Analyzer")
st.markdown(
    "Analyze Reddit comments in real-time to detect whether the tone is **Positive**, **Neutral**, or **Negative**."
)

# tab1, tab2 = st.tabs(["📝 Single Comment Analysis", "📊 Batch File Analysis"])

# ---------------------------------------------------------
# Tab 1: Single Comment Analysis
# ---------------------------------------------------------
# with tab1:
    # st.subheader("Analyze an Individual Comment")
st.header("Analyze an Individual Comment")
comment_text = st.text_area(
    "Enter Reddit comment text:",
    height=140,
    placeholder="Type or paste a comment here...",
)

predict_btn = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)

if predict_btn:
    if not comment_text.strip():
        st.warning("⚠️ Please enter some comment text to classify.")
    elif not server_online:
        st.error("❌ Cannot connect to FastAPI server. Please ensure the backend is running.")
    else:
        with st.spinner("Analyzing sentiment..."):
            try:
                response = requests.post(
                    f"{api_url.rstrip('/')}/predict",
                    json={"text": comment_text},
                    timeout=5,
                )
                
                if response.status_code == 200:
                    result = response.json()
                    sentiment = result.get("sentiment", "").title()

                    badge_class = "badge-neutral"
                    if "pos" in sentiment.lower():
                        badge_class = "badge-positive"
                    elif "neg" in sentiment.lower():
                        badge_class = "badge-negative"

                    st.markdown("### Prediction Result")
                    st.markdown(
                        f'<div class="sentiment-badge {badge_class}">{sentiment}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

# # ---------------------------------------------------------
# # Tab 2: Batch Analysis (CSV)
# # ---------------------------------------------------------
# with tab2:
#     st.subheader("Batch Process Comments from CSV")
#     st.markdown("Upload a CSV file containing comments to classify them simultaneously.")

#     uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

#     if uploaded_file is not None:
#         try:
#             df = pd.read_csv(uploaded_file)
#             st.write("Preview of uploaded data:")
#             st.dataframe(df.head(5), use_container_width=True)

#             text_col = st.selectbox("Select the column containing comments:", df.columns)

#             if st.button("⚡ Run Batch Prediction", type="primary"):
#                 if not server_online:
#                     st.error("❌ Cannot connect to FastAPI backend.")
#                 else:
#                     progress_bar = st.progress(0)
#                     status_text = st.empty()

#                     sentiments = []
#                     total = len(df)
                    
#                     for i, row in df.iterrows():
#                         text_val = str(row[text_col])
#                         try:
#                             res = requests.post(
#                                 f"{api_url.rstrip('/')}/predict",
#                                 json={"text": text_val},
#                                 timeout=5,
#                             )
#                             if res.status_code == 200:
#                                 r_json = res.json()
#                                 sentiments.append(r_json.get("sentiment", "").title())
#                             else:
#                                 sentiments.append("Error")
#                         except Exception:
#                             sentiments.append("Failed")

#                         progress_bar.progress((i + 1) / total)
#                         status_text.text(f"Processed {i + 1} / {total} comments")

#                     df["predicted_sentiment"] = sentiments

#                     st.success("🎉 Batch processing complete!")

#                     st.subheader("Sentiment Distribution")
#                     sentiment_counts = df["predicted_sentiment"].value_counts()
#                     st.bar_chart(sentiment_counts)

#                     st.dataframe(df, use_container_width=True)
#                     csv_data = df.to_csv(index=False).encode("utf-8")
#                     st.download_button(
#                         label="📥 Download Results CSV",
#                         data=csv_data,
#                         file_name="sentiment_predictions.csv",
#                         mime="text/csv",
#                     )
#         except Exception as err:
#             st.error(f"Failed to read CSV: {err}")