import joblib
import os
import re

MODEL_PATH = "models/pipeline.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found")

model = joblib.load(MODEL_PATH)


PHISHING_TYPE_RULES = {
    "credential harvesting": [
        "verify", "login", "password", "account", "sign in", "credentials"
    ],
    "financial scam": [
        "invoice", "payment", "bank", "transfer", "refund", "tax"
    ],
    "urgency": [
        "urgent", "immediately", "action required", "asap", "final notice"
    ],
    "authority scam": [
        "ceo", "manager", "it department", "admin", "security team"
    ],
    "romance scam": [
        "love", "dear", "relationship", "dating", "gift"
    ],
}
SUSPICIOUS_WORDS = [
    "urgent",
    "immediately",
    "verify",
    "suspended",
    "action required",
    "click",
    "confirm",
    "account",
    "security alert"
]

def detect_phishing_type(text: str, links: list, urgency_words: list) -> str:
    text_lower = text.lower()

    if links:
        for link in links:
            if link.startswith("http://") or link.count(".") < 2:
                return "credential_harvesting"

    if urgency_words:
        return "urgency"

    for p_type, keywords in PHISHING_TYPE_RULES.items():
        for kw in keywords:
            if kw in text_lower:
                return p_type

    return "generic phishing"


def predict_email(text: str) -> float:
    prob = model.predict_proba([text])[0]
    return float(prob[1])

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
    url_pattern = r"http[s]?://\S+|www\.\S+"
    return re.findall(url_pattern, text)

def detect_urgency_words(text: str) -> list:

    text_lower = text.lower()
    found = [word for word in SUSPICIOUS_WORDS if word in text_lower]
    return found
