# -*- coding: utf-8 -*-

from __future__ import annotations
import os, sqlite3
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

def create_database(db_path: str = "data/training_data.db") -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS vulnerabilities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            label TEXT NOT NULL
        )
    """)
    conn.commit()
    cur.execute("SELECT COUNT(*) FROM vulnerabilities")
    count = cur.fetchone()[0]
    if count == 0:
        samples = [
            ("<script>alert('x')</script>", "xss"),
            ("<img src=x onerror=alert(1)>", "xss"),
            ("<svg/onload=alert(1)>", "xss"),
            ("\" OR \"1\"=\"1\"--", "sqli"),
            ("' OR '1'='1'--", "sqli"),
            ("SELECT * FROM users WHERE name = '' OR 1=1;", "sqli"),
            ("http://localhost:8000/admin", "ssrf"),
            ("file:///etc/passwd", "ssrf"),
            ("gopher://127.0.0.1:3306/_", "ssrf"),
            ("https://example.com/user/1", "idor"),
            ("https://example.com/account/../../etc/passwd", "idor"),
            ("../admin/config", "idor"),
        ]
        cur.executemany(
            "INSERT INTO vulnerabilities (text, label) VALUES (?, ?)", samples
        )
        conn.commit()
    conn.close()

def load_data(db_path: str = "data/training_data.db") -> Tuple[List[str], List[str]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT text, label FROM vulnerabilities")
    records = cur.fetchall()
    conn.close()
    return [row[0] for row in records], [row[1] for row in records]

def train_and_save_model(db_path: str = "data/training_data.db", model_path: str = "data/vuln_classifier.joblib", vectorizer_path: str = "data/vectorizer.joblib") -> float:
    texts, labels = load_data(db_path)
    if not texts or not labels:
        raise ValueError("Dataset is empty; please populate the database first.")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    classifier = LogisticRegression(max_iter=200)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.2f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(classifier, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    return acc

if __name__ == "__main__":
    create_database()
    accuracy = train_and_save_model()
    print(f"Model trained and saved with accuracy: {accuracy:.2f}")
