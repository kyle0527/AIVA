import os
import sqlite3
from typing import Any

# 使用統一的可選依賴管理器
try:
    from utilities.optional_deps import deps, has_sklearn, require_sklearn
except ImportError:
    # 如果工具模組不可用，回退到基本檢查
    import importlib.util
    import logging
    
    sklearn_spec = importlib.util.find_spec("sklearn")
    joblib_spec = importlib.util.find_spec("joblib")
    
    def has_sklearn() -> bool:
        return sklearn_spec is not None and joblib_spec is not None
    
    def require_sklearn():
        if not has_sklearn():
            raise ImportError(
                "scikit-learn and joblib are required for model training. "
                "Install with: pip install scikit-learn joblib"
            )
        import sklearn
        import joblib
        return sklearn, joblib

# 條件導入機器學習組件
if has_sklearn():
    try:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        # 即使檢測到也可能導入失敗，提供降級處理
        import logging
        logging.warning(f"sklearn/joblib detected but import failed: {e}")
        # 設置 None 占位符用於運行時檢查
        joblib = None
        TfidfVectorizer = None
        LogisticRegression = None
        accuracy_score = None
        classification_report = None
        train_test_split = None
else:
    # 提供 None 占位符以避免 NameError
    joblib = None
    TfidfVectorizer = None
    LogisticRegression = None
    accuracy_score = None
    classification_report = None
    train_test_split = None


def create_database(db_path: str = "data/training_data.db") -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS vulnerabilities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            label TEXT NOT NULL
        )
    """
    )
    conn.commit()
    cur.execute("SELECT COUNT(*) FROM vulnerabilities")
    count = cur.fetchone()[0]
    if count == 0:
        samples = [
            ("<script>alert('x')</script>", "xss"),
            ("<img src=x onerror=alert(1)>", "xss"),
            ("<svg/onload=alert(1)>", "xss"),
            ('" OR "1"="1"--', "sqli"),
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


def load_data(db_path: str = "data/training_data.db") -> tuple[list[str], list[str]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT text, label FROM vulnerabilities")
    records = cur.fetchall()
    conn.close()
    return [row[0] for row in records], [row[1] for row in records]


def train_and_save_model(
    db_path: str = "data/training_data.db",
    model_path: str = "data/vuln_classifier.joblib",
    vectorizer_path: str = "data/vectorizer.joblib",
) -> float:
    # 運行時檢查：根據網路最佳實踐提供清晰的錯誤信息
    if not has_sklearn():
        require_sklearn()  # 這會拋出詳細的錯誤信息
    
    texts, labels = load_data(db_path)
    if not texts or not labels:
        raise ValueError("Dataset is empty; please populate the database first.")
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
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


def load_model(
    model_path: str = "data/vuln_classifier.joblib",
    vectorizer_path: str = "data/vectorizer.joblib"
) -> tuple[Any, Any]:
    """載入訓練好的模型和向量化器"""
    if not has_sklearn():
        require_sklearn()  # 這會拋出詳細的錯誤信息
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Model files not found. Please train the model first.")
    
    classifier = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return classifier, vectorizer


if __name__ == "__main__":
    create_database()
    accuracy = train_and_save_model()
    print(f"Model trained and saved with accuracy: {accuracy:.2f}")
