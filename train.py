"""
Phishing Email Detector - Model Training Script
Trains a Logistic Regression model with TF-IDF features for phishing detection.
"""

from src.data_preparation import load_data, clean_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import pandas as pd
import numpy as np

# Configuration
DATA_PATH = "data/raw/phishing_email_dataset.csv"
MODEL_PATH = "models/pipeline.joblib"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Load data
print("Loading dataset...")
df = load_data(DATA_PATH)
df["clean_text"] = clean_text(df["raw_text"])

# Balance dataset
print("Balancing dataset...")
df_phishing = df[df["Spam Label"] == 1]
df_safe = df[df["Spam Label"] == 0]

min_size = min(len(df_phishing), len(df_safe))

df_phishing_balanced = resample(
    df_phishing,
    n_samples=min_size,
    random_state=RANDOM_STATE
)
df_safe_balanced = resample(
    df_safe,
    n_samples=min_size,
    random_state=RANDOM_STATE
)

df_balanced = pd.concat([df_phishing_balanced, df_safe_balanced])
df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print(f"Dataset balanced: {len(df_balanced)} total samples (50% phishing, 50% safe)")

# Prepare features and labels
X = df_balanced["clean_text"]
y = df_balanced["Spam Label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Build pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=3000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.85,
        sublinear_tf=True
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        C=0.5,
        l1_ratio=0,
        solver='lbfgs',
        random_state=RANDOM_STATE
    ))
])

# Train model
print("\nTraining model...")
pipeline.fit(X_train, y_train)

# Evaluate model
print("\nEvaluating model performance...")
train_acc = accuracy_score(y_train, pipeline.predict(X_train))
y_pred = pipeline.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

if train_acc - test_acc > 0.05:
    print("Warning: Potential overfitting detected (train-test accuracy gap > 5%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                Predicted Safe  Predicted Phishing")
print(f"Actual Safe:    {cm[0][0]:14d}  {cm[0][1]:18d}")
print(f"Actual Phishing:{cm[1][0]:14d}  {cm[1][1]:18d}")

# Save model
print(f"\nSaving model to {MODEL_PATH}...")
joblib.dump(pipeline, MODEL_PATH)
print("Model saved successfully.")

# Test with sample emails
print("\n" + "=" * 80)
print("Testing model with sample emails:")
print("=" * 80)

test_cases = [
    ("Verify your account immediately or it will be suspended!", "phishing"),
    ("Hi John, the meeting is scheduled for 3pm tomorrow.", "safe"),
    ("URGENT: Your payment is overdue. Click here now!", "phishing"),
    ("Thanks for your email. I'll review the document and get back to you.", "safe"),
    ("You've won $1,000,000! Claim your prize now!", "phishing"),
    ("Can you send me the quarterly report by end of day?", "safe"),
    ("Your package is waiting. Confirm delivery address here.", "phishing"),
    ("Let's schedule a call next week to discuss the project.", "safe"),
]

model = joblib.load(MODEL_PATH)
correct_predictions = 0

for text, expected_label in test_cases:
    prob = model.predict_proba([text])[0]
    predicted_label = "safe" if prob[0] > prob[1] else "phishing"
    is_correct = (predicted_label == expected_label)
    correct_predictions += is_correct

    status = "[CORRECT]" if is_correct else "[INCORRECT]"
    print(f"\n{status} Predicted: {predicted_label.upper()} (Expected: {expected_label.upper()})")
    print(f"Text: {text[:70]}")
    print(f"Confidence: Safe={prob[0]:.2%}, Phishing={prob[1]:.2%}")

print("\n" + "=" * 80)
print(f"Sample Test Accuracy: {correct_predictions}/{len(test_cases)} ({correct_predictions / len(test_cases):.1%})")
print("=" * 80)

print("\nTraining complete. Model ready for deployment.")
