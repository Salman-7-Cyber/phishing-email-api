from src.data_preparation import load_data,clean_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

DATA_PATH = "data/raw/phishing_email_dataset.csv"


df = load_data(DATA_PATH)

df["clean_text"] = clean_text(df["raw_text"])

X = df["clean_text"]
y = df["Spam Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify = y
)

tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000,
)

Pipeline = Pipeline([
    ("tfidf_vectorizer", TfidfVectorizer(
        stop_words='english',
        max_features=10000,
        ngram_range=(1, 2),
    )),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    ))
])

Pipeline.fit(X_train, y_train)
y_pred = Pipeline.predict(X_test)

joblib.dump(Pipeline, "models/pipeline.joblib")
