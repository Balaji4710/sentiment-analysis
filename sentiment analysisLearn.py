import pandas as pd
import re
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import VotingClassifier
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    Trainer, TrainingArguments
)
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def clean_text(text):
    text = text.lower()
    contractions = {
        "n't": " not", "'re": " are", "'s": " is", "'ll": " will",
        "'ve": " have", "'m": " am"
    }
    for k, v in contractions.items():
        text = text.replace(k, v)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels':         torch.tensor(int(self.labels[idx]), dtype=torch.long)
        }
if __name__ == '__main__':

    print(f"Using device: {device}")
    if device.type == "cpu":
        print("WARNING: No GPU detected. Training will be slow.")
        print("TIP: Use Google Colab (free T4 GPU) for 10x speedup.\n")
    data = pd.read_csv("C:/Users/Home/Downloads/archive/Train.csv")
    df = pd.DataFrame(data)
    df['cleaned_text'] = df['text'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['label'], test_size=0.2, random_state=42
    )
    QUICK_TEST_MODE = False
    if QUICK_TEST_MODE:
        sample_size = int(len(X_train) * 0.2)
        X_train = X_train[:sample_size]
        y_train = y_train[:sample_size]
        print(f"Quick test mode: using {sample_size} samples\n")

    models = {
        "Logistic Regression": LogisticRegression(C=1.0, solver='liblinear', class_weight='balanced'),
        "SVM":                 LinearSVC(C=1.0, class_weight='balanced', max_iter=2000),
        "Naive Bayes":         MultinomialNB(alpha=1.0)
    }

    trained_pipelines = {}
    for name, clf in models.items():
        print(f"--- Training {name} ---")
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ('clf', clf)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        trained_pipelines[name] = pipeline
        joblib.dump(pipeline, f"sentiment_model_{name.lower().replace(' ', '_')}.pkl")
    print("\n--- Training Voting Classifier Ensemble ---")
    ensemble = VotingClassifier(
        estimators=[
            ('lr',  models["Logistic Regression"]),
            ('svm', models["SVM"]),
            ('nb',  models["Naive Bayes"])
        ],
        voting='hard'
    )
    ensemble_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ('clf', ensemble)
    ])
    ensemble_pipeline.fit(X_train, y_train)
    y_pred_ensemble = ensemble_pipeline.predict(X_test)
    print(f"Ensemble Accuracy: {accuracy_score(y_test, y_pred_ensemble):.2%}")
    print(classification_report(y_test, y_pred_ensemble))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_ensemble))
    joblib.dump(ensemble_pipeline, "sentiment_model_ensemble.pkl")
    print("\n--- Training DistilBERT ---")

    tokenizer     = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_dataset = SentimentDataset(X_train.tolist(), y_train.tolist(), tokenizer)
    test_dataset  = SentimentDataset(X_test.tolist(),  y_test.tolist(),  tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=2
    )
    for param in model.distilbert.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    training_args = TrainingArguments(
        output_dir='./distilbert_model',
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir='./logs_distilbert',
        logging_steps=50,
        use_cpu=not torch.cuda.is_available(),
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,      # Windows FIX: must be 0 on Windows
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()
    trainer.save_model('./distilbert_model')
    tokenizer.save_pretrained('./distilbert_model')
    print("DistilBERT saved to ./distilbert_model")
    y_pred_bert = []
    model.eval()
    model.to(device)
    loader = DataLoader(test_dataset, batch_size=64, num_workers=0)
    for batch in loader:
        with torch.no_grad():
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            preds          = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            y_pred_bert.extend(preds)

    print(f"\nDistilBERT Accuracy: {accuracy_score(y_test, y_pred_bert):.2%}")
    print(classification_report(y_test, y_pred_bert))

   
    print("\n--- Loading models for inference ---")
    classical_model      = joblib.load("sentiment_model_ensemble.pkl")
    distilbert_model     = DistilBertForSequenceClassification.from_pretrained(
                               "./distilbert_model").eval().to(device)
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained("./distilbert_model")

    def predict_unified(review_text):
        classical_pred = classical_model.predict([clean_text(review_text)])[0]

        inputs = distilbert_tokenizer(
            review_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = distilbert_model(**inputs)
        distilbert_pred = torch.argmax(outputs.logits, dim=1).item()

        votes      = [classical_pred, distilbert_pred]
        final_pred = 1 if votes.count(1) >= 1 else 0

        print(f"  Classical ensemble : {'Positive' if classical_pred == 1 else 'Negative'}")
        print(f"  DistilBERT         : {'Positive' if distilbert_pred == 1 else 'Negative'}")
        return "Positive" if final_pred == 1 else "Negative"
    print("\nEnter reviews to classify (type 'exit' to quit):")
    while True:
        user_review = input("\nReview: ").strip()
        if user_review.lower() == "exit":
            print("Exiting. Goodbye!")
            break
        if not user_review:
            print("Please enter a review.")
            continue
        result = predict_unified(user_review)
        print(f"  Final Sentiment    : {result}")