from sentiment_model import clean_text, load_dataset, predict_sentiment, train_and_compare


def test_clean_text_removes_urls_and_normalizes_space():
    assert clean_text(" Great!!! https://example.com ") == "great"


def test_training_and_prediction():
    data = load_dataset()
    model, results, metadata = train_and_compare(data)
    label, confidence = predict_sentiment(model, "excellent quality and helpful service")
    assert label == "Positive"
    assert 0.5 <= confidence <= 1
    assert len(results) == 2
    assert "accuracy" in metadata
