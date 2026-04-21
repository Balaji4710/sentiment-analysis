import os
import re
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
DATA_PATH = "C:/Users/Home/Downloads/archive/Train.csv"
RANDOM_STATE = 42
MODEL_FILE = "best_sentiment_model.pkl"
METADATA_FILE = "model_metadata.joblib"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()
def train_and_select_best(data_path: str):
    log.info("Reading dataset...")
    df = pd.read_csv(data_path).dropna(subset=["text", "label"]).reset_index(drop=True)
    df["cleaned_text"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"], df["label"], test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
    )
    models = {
        "Logistic Regression": LogisticRegression(C=1.0, solver="liblinear"),
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "SVM (LinearSVC)": LinearSVC(C=1.0, max_iter=2000)
    }
    best_accuracy = 0.0
    best_pipeline = None
    best_model_name = ""
    log.info("Starting model training...")
    for name, clf in models.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("clf", clf)
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        
        log.info(f"-> {name} achieved: {acc:.2%}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_pipeline = pipeline
            best_model_name = name

    log.info(f"Best model: {best_model_name} with {best_accuracy:.2%} accuracy.")
    joblib.dump(best_pipeline, MODEL_FILE)
    joblib.dump({"name": best_model_name, "accuracy": best_accuracy}, METADATA_FILE)
    
    return best_pipeline, best_model_name


if __name__ == "__main__":
    if not os.path.exists(MODEL_FILE) or not os.path.exists(METADATA_FILE):
        if not os.path.exists(DATA_PATH):
            log.error(f"File not found: {DATA_PATH}")
            exit(1)
        model, model_name = train_and_select_best(DATA_PATH)
    else:
        log.info("Loading the previously determined best model...")
        model = joblib.load(MODEL_FILE)
        metadata = joblib.load(METADATA_FILE)
        model_name = metadata['name']
        log.info(f"Currently using: {model_name} (Acc: {metadata['accuracy']:.2%})")
    print("\n" + "="*50)
    print(f"PREDICTOR ACTIVE: Using {model_name}")
    print("Type 'exit' to quit.")
    print("="*50)
    while True:
        user_input = input("\nEnter text to analyze: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input:
            continue
        cleaned = clean_text(user_input)
        pred = model.predict([cleaned])[0]
        label = "Positive" if pred == 1 else "Negative"
        color = "\033[92m" if label == "Positive" else "\033[91m"
        
        print(f"Analysis: {color}{label}\033[0m")