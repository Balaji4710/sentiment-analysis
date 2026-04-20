import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s!]", "", text)
    text = text.replace("n't", " not").replace("not ", "not_")
    return text
data = pd.read_csv("C:/Users/Home/Downloads/archive/Train.csv")
df = pd.DataFrame(data)
df['cleaned_text'] = df['text'].apply(clean_text)
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['label'], test_size=0.2, random_state=42
)
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words='english'
    )),
    ('clf', LogisticRegression(C=1.0, solver='liblinear', class_weight='balanced'))
])
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)
print("\nModel Evaluation on Test Set:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
def predict_sentiment(review_text):
    sample_cleaned = [clean_text(review_text)]
    prediction = model_pipeline.predict(sample_cleaned)
    probability = model_pipeline.predict_proba(sample_cleaned)
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    confidence = probability.max()
    return sentiment, confidence
print("\nEnter reviews to classify sentiment (type 'exit' to quit):")
while True:
    user_review = input("Review: ")
    if user_review.lower() == "exit":
        break
    sentiment, confidence = predict_sentiment(user_review)
    print(f"Sentiment: {sentiment} | Confidence: {confidence:.2%}\n")

joblib.dump(model_pipeline, "sentiment_model.pkl")
