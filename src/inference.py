import joblib
import re
import os
from typing import Optional

MODEL_PATH = os.path.join("models", "pipeline.joblib")
model: Optional[object] = None


def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
    return model


# Phishing type detection rules
PHISHING_TYPE_RULES = {
    "credential_harvesting": [
        "verify", "login", "password", "account", "sign in", "credentials"
    ],
    "financial_scam": [
        "invoice", "payment", "bank", "transfer", "refund", "tax", "billing"
    ],
    "urgency": [
        "urgent", "immediately", "action required", "asap", "final notice", "expires"
    ],
    "authority_impersonation": [
        "ceo", "manager", "it department", "admin", "security team", "support"
    ],
    "prize_scam": [
        "won", "winner", "prize", "congratulations", "claim", "lottery"
    ],
}

SUSPICIOUS_WORDS = [
    "urgent", "immediately", "verify", "suspended", "action required",
    "click", "confirm", "account", "security alert", "expires",
    "limited time", "act now", "suspended", "locked"
]


def detect_phishing_type(text: str, links: list, urgency_words: list) -> str:
    text_lower = text.lower()

    # Check for suspicious links
    if links:
        for link in links:
            # HTTP instead of HTTPS or unusual domain structure
            if link.startswith("http://") or link.count(".") < 2:
                return "credential_harvesting"

    # Check for urgency tactics
    if len(urgency_words) >= 2:
        return "urgency"

    # Check against phishing type rules
    for p_type, keywords in PHISHING_TYPE_RULES.items():
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches >= 2:  # Require at least 2 keyword matches
            return p_type

    return "generic_phishing"


def predict_email(text: str) -> float:
    if not text or len(text.strip()) < 5:
        raise ValueError("Text too short for analysis")

    clf = load_model()
    prob = clf.predict_proba([text])[0]
    return prob[1]  # Probability of phishing class


def get_risk_message(score: float) -> dict:
    if score >= 0.70:
        return {
            "level": "High Risk (Phishing)",
            "explanation": (
                "This email strongly matches known phishing patterns. "
                "It may contain deceptive language or malicious intent."
            ),
            "recommendation": (
                "Do NOT click any links or download attachments. "
                "Report or delete this email immediately."
            ),
        }
    elif score >= 0.40:
        return {
            "level": "Medium Risk (Suspicious)",
            "explanation": (
                "This email shares some characteristics with phishing messages. "
                "Caution is advised."
            ),
            "recommendation": (
                "Avoid clicking links or sharing sensitive information. "
                "Verify the sender through a trusted channel."
            ),
        }
    else:
        return {
            "level": "Low Risk (Likely Safe)",
            "explanation": (
                "This email does not show common phishing indicators "
                "based on language and structure."
            ),
            "recommendation": (
                "No immediate threat detected. "
                "You may proceed normally while staying cautious."
            ),
        }


def detect_links(text: str) -> list:
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    return re.findall(url_pattern, text, re.IGNORECASE)


def detect_urgency_words(text: str) -> list:
    text_lower = text.lower()
    found = [word for word in SUSPICIOUS_WORDS if word in text_lower]
    return found
