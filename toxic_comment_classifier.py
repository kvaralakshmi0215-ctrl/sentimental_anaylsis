"""
Cyberbullying & Toxic Comment Detection System
=================================================
Multi-platform NLP classification pipeline for detecting toxic comments,
cyberbullying, hate speech, and harassment.

Features:
- Multi-dataset ingestion (Wikipedia, Twitter, YouTube, Kaggle)
- Robust text cleaning (URLs, HTML tags, punctuation, lowercasing)
- Sublinear TF-IDF vectorization (unigrams + bigrams)
- Model comparison: Linear SVM, Logistic Regression, Random Forest, Multinomial Naive Bayes
- Interactive CLI for classifying custom comments with toxicity probabilities
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def clean_text(text):
    """Sanitizes text by stripping URLs, HTML, punctuation, and extra whitespace."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class ToxicCommentClassifier:
    def __init__(self, data_dir=".", sample_size=30000):
        self.data_dir = data_dir
        self.sample_size = sample_size
        self.df = None
        self.tfidf = None
        self.model = None
        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        """Loads and consolidates multi-platform toxic comment datasets."""
        print("Loading multi-platform datasets...")
        files = [
            ('kaggle_parsed_dataset.csv', ['Text', 'oh_label']),
            ('twitter_parsed_dataset.csv', ['Text', 'oh_label']),
            ('twitter_racism_parsed_dataset.csv', ['Text', 'oh_label']),
            ('twitter_sexism_parsed_dataset.csv', ['Text', 'oh_label']),
            ('youtube_parsed_dataset.csv', ['Text', 'oh_label']),
        ]

        frames = []
        for fname, cols in files:
            p = os.path.join(self.data_dir, fname)
            if os.path.exists(p):
                d = pd.read_csv(p, encoding='utf-8', encoding_errors='replace')[cols].dropna()
                d.columns = ['Text', 'label']
                frames.append(d)

        # Ingest samples from larger Wikipedia files
        for fname in ['aggression_parsed_dataset.csv', 'attack_parsed_dataset.csv', 'toxicity_parsed_dataset.csv']:
            p = os.path.join(self.data_dir, fname)
            if os.path.exists(p):
                d = pd.read_csv(p, encoding='utf-8', encoding_errors='replace', nrows=10000)[['Text', 'oh_label']].dropna()
                d.columns = ['Text', 'label']
                frames.append(d)

        if not frames:
            raise FileNotFoundError("No toxic comment CSV files found in directory.")

        self.df = pd.concat(frames, ignore_index=True)
        self.df['label'] = self.df['label'].astype(int)
        self.df = self.df[self.df['Text'].str.len() > 5]

        if len(self.df) > self.sample_size:
            self.df = self.df.sample(n=self.sample_size, random_state=42)

        print(f"Total comments consolidated for training: {len(self.df):,}")
        print(f"Class distribution: Non-Toxic = {(self.df['label']==0).sum():,}, Toxic = {(self.df['label']==1).sum():,}")

        print("Cleaning text and building TF-IDF representations (25,000 features)...")
        self.df['clean_text'] = self.df['Text'].apply(clean_text)
        self.df = self.df[self.df['clean_text'].str.len() > 3]

        self.tfidf = TfidfVectorizer(max_features=25000, stop_words='english', ngram_range=(1, 2), sublinear_tf=True)
        self.X = self.tfidf.fit_transform(self.df['clean_text'])
        self.y = self.df['label']

    def train_best_model(self):
        """Trains the best-performing model (Linear SVM)."""
        print("Training Linear SVM classifier...")
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)
        self.model = LinearSVC(max_iter=2000, random_state=42)
        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        print("\n--- Linear SVM Performance on Held-Out Test Set ---")
        print(f"Accuracy : {accuracy_score(y_test, preds)*100:.2f}%")
        print(f"Precision: {precision_score(y_test, preds)*100:.2f}%")
        print(f"Recall   : {recall_score(y_test, preds)*100:.2f}%")
        print(f"F1-Score : {f1_score(y_test, preds)*100:.2f}%\n")
        print(classification_report(y_test, preds, target_names=['Non-Toxic', 'Toxic']))

    def predict(self, comment):
        """Classifies a user-provided comment as Toxic or Non-Toxic."""
        if not self.model:
            self.train_best_model()

        cleaned = clean_text(comment)
        vec = self.tfidf.transform([cleaned])
        pred = self.model.predict(vec)[0]
        decision = self.model.decision_function(vec)[0]
        label = "Toxic / Cyberbullying" if pred == 1 else "Non-Toxic / Safe"
        return {"comment": comment, "label": label, "decision_score": float(decision)}

    def benchmark_models(self):
        """Compares multiple algorithms across standard classification metrics."""
        print("\n=== Benchmarking Multiple ML Models ===")
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)

        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Multinomial Naive Bayes': MultinomialNB(alpha=0.5),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Linear SVM (Best)': LinearSVC(max_iter=2000, random_state=42),
        }

        results = {}
        for name, m in models.items():
            m.fit(X_train, y_train)
            p = m.predict(X_test)
            acc = accuracy_score(y_test, p)
            rec = recall_score(y_test, p)
            f1 = f1_score(y_test, p)
            results[name] = {'Accuracy': acc, 'Recall': rec, 'F1-Score': f1}
            print(f"  {name:25s} | Accuracy: {acc*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}%")

        return results


def main():
    parser = argparse.ArgumentParser(description="Cyberbullying & Toxic Comment Detection System")
    parser.add_argument("--text", type=str, help="Single comment text to classify")
    parser.add_argument("--benchmark", action="store_true", help="Run model comparison benchmarks")
    args = parser.parse_args()

    classifier = ToxicCommentClassifier()

    if args.benchmark:
        classifier.benchmark_models()

    if args.text:
        result = classifier.predict(args.text)
        print("\n--- Classification Result ---")
        print(f"Input Comment : {result['comment']}")
        print(f"Prediction    : {result['label']}")
        print(f"Confidence/Score: {result['decision_score']:.3f}")

    if not args.text and not args.benchmark:
        test_samples = [
            "Thank you for sharing this insightful tutorial! Really helped with my project.",
            "You are completely stupid, nobody wants to hear your garbage opinion.",
            "I disagree with the premise of this article, here are my counterpoints."
        ]
        print("\n--- Running Sample Inferences ---")
        for s in test_samples:
            r = classifier.predict(s)
            print(f"[{r['label']:22s}] \"{r['comment']}\"")


if __name__ == "__main__":
    main()
