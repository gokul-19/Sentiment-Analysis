import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sentiment_model import DEFAULT_DATASET, clean_text, load_dataset, load_or_train_model, predict_sentiment, train_and_compare

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="wide")


@st.cache_resource
def get_model():
    return load_or_train_model()


@st.cache_data
def get_data():
    return load_dataset(DEFAULT_DATASET)


@st.cache_data
def get_evaluation():
    _, comparison, metadata = train_and_compare(get_data())
    return comparison, metadata


def top_terms_by_sentiment(data: pd.DataFrame, limit: int = 10) -> tuple[pd.Series, pd.Series]:
    """Find discriminative terms using mean class-specific TF-IDF weights."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(data.text)
    terms = pd.Index(vectorizer.get_feature_names_out())
    # SciPy sparse matrices require a NumPy boolean array here; a pandas
    # Series works in some versions but fails in newer SciPy releases.
    positive_mask = data.label.eq(1).to_numpy()
    negative_mask = data.label.eq(0).to_numpy()
    positive = pd.Series(matrix[positive_mask].mean(axis=0).A1, index=terms).nlargest(limit).sort_values()
    negative = pd.Series(matrix[negative_mask].mean(axis=0).A1, index=terms).nlargest(limit).sort_values()
    return positive, negative


st.title("💬 Sentiment Analysis System")
st.caption("Classify review text as positive or negative using a TF-IDF machine-learning model.")

model = get_model()
data = get_data()
positive_share = (data.label == 1).mean()
col1, col2, col3 = st.columns(3)
col1.metric("Reviews", len(data))
col2.metric("Positive", f"{positive_share:.0%}")
col3.metric("Negative", f"{1 - positive_share:.0%}")

tab_predict, tab_batch, tab_dashboard = st.tabs(["Analyze text", "Batch analysis", "Dashboard"])

with tab_predict:
    text = st.text_area("Paste a review", placeholder="The product quality is excellent and delivery was fast!", height=130)
    if st.button("Analyze sentiment", type="primary"):
        try:
            label, confidence = predict_sentiment(model, text)
            st.success(f"{label.upper()} sentiment — {confidence:.1%} confidence")
            st.caption(f"Preprocessed text: {clean_text(text)}")
        except ValueError as error:
            st.warning(str(error))

with tab_batch:
    st.write("Upload a CSV containing a `text` column. The result can be downloaded after analysis.")
    upload = st.file_uploader("CSV file", type="csv")
    if upload is not None:
        try:
            batch = pd.read_csv(upload)
            if "text" not in batch.columns:
                raise ValueError("The uploaded CSV needs a column named 'text'.")
            probabilities = model.predict_proba(batch.text.fillna("").map(clean_text))[:, 1]
            batch["sentiment"] = pd.Series(probabilities).ge(0.5).map({True: "Positive", False: "Negative"})
            batch["positive_probability"] = probabilities.round(4)
            st.dataframe(batch, use_container_width=True)
            st.download_button("Download results", batch.to_csv(index=False).encode("utf-8"), "sentiment_results.csv", "text/csv")
        except Exception as error:
            st.error(f"Could not analyze the file: {error}")

with tab_dashboard:
    comparison, evaluation = get_evaluation()
    counts = data.label.map({0: "Negative", 1: "Positive"}).value_counts().reindex(["Positive", "Negative"])
    data = data.assign(review_length=data.text.str.split().str.len())
    left, right = st.columns(2)
    with left:
        figure, axis = plt.subplots(figsize=(6, 3.5))
        axis.bar(counts.index, counts.values, color=["#2e8b57", "#d1495b"])
        axis.set(title="Sentiment distribution", xlabel="Sentiment", ylabel="Review count")
        st.pyplot(figure, clear_figure=True)
    with right:
        figure, axis = plt.subplots(figsize=(6, 3.5))
        for label, color in [(1, "#2e8b57"), (0, "#d1495b")]:
            axis.hist(data.loc[data.label.eq(label), "review_length"], bins=8, alpha=.65, label={1: "Positive", 0: "Negative"}[label], color=color)
        axis.set(title="Review-length distribution", xlabel="Words per review", ylabel="Count")
        axis.legend()
        st.pyplot(figure, clear_figure=True)

    left, right = st.columns(2)
    with left:
        figure, axis = plt.subplots(figsize=(6, 3.5))
        axis.bar(comparison.Model, comparison["F1 score"], color="#3867d6")
        axis.set_ylim(0, 1)
        axis.set(title="Model comparison (F1 score)", xlabel="", ylabel="F1 score")
        axis.tick_params(axis="x", rotation=12)
        st.pyplot(figure, clear_figure=True)
    with right:
        matrix = evaluation["confusion_matrix"]
        figure, axis = plt.subplots(figsize=(5.5, 3.5))
        image = axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, matrix[row][column], ha="center", va="center")
        axis.set(title=f"Confusion matrix — {evaluation['best_model']}", xticks=[0, 1], yticks=[0, 1], xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"], xlabel="Predicted", ylabel="Actual")
        figure.colorbar(image, ax=axis)
        st.pyplot(figure, clear_figure=True)

    st.subheader("Most distinctive terms")
    positive_terms, negative_terms = top_terms_by_sentiment(data)
    left, right = st.columns(2)
    with left:
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.barh(positive_terms.index, positive_terms.values, color="#2e8b57")
        axis.set(title="Positive terms", xlabel="Mean TF-IDF weight")
        st.pyplot(figure, clear_figure=True)
    with right:
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.barh(negative_terms.index, negative_terms.values, color="#d1495b")
        axis.set(title="Negative terms", xlabel="Mean TF-IDF weight")
        st.pyplot(figure, clear_figure=True)

    with st.expander("View labeled training data"):
        st.dataframe(data[["text", "label", "source", "review_length"]].assign(label=lambda d: d.label.map({0: "Negative", 1: "Positive"})), use_container_width=True)
