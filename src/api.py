from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# استيراد الدوال الجاهزة (العقل)
from src.inference import (
    predict_email,
    get_risk_message,
    detect_links,
    detect_urgency_words,
    detect_phishing_type,
)

app = FastAPI()


# ====== شكل الطلب ======
class AnalyzeRequest(BaseModel):
    text: str


# ====== اختبار ======
@app.get("/")
def root():
    return {"status": "API is running"}


# ====== التحليل الحقيقي ======
@app.post("/analyze")
def analyze_email(request: AnalyzeRequest):
    try:
        text = request.text

        if len(text.split()) < 5:
            return {"error": "Not enough text to analyze"}

        score = predict_email(text)
        risk_info = get_risk_message(score)

        links = detect_links(text)
        urgent_words = detect_urgency_words(text)

        reasons = []
        if links:
            reasons.append("Suspicious link detected")
        else:
            reasons.append("No suspicious links found")

        if urgent_words:
            reasons.append(
                f"Urgent or pressure language detected ({', '.join(urgent_words)})"
            )
        else:
            reasons.append("No urgent or threatening language detected")

        phishing_type = "legitimate"
        if score >= 0.40:
            phishing_type = detect_phishing_type(text, links, urgent_words)

        return {
            "risk_score": round(score, 4),
            "risk_category": (
                "high" if score >= 0.70 else
                "medium" if score >= 0.40 else
                "low"
            ),
            "risk_level": risk_info["level"],
            "explanation": risk_info["explanation"],
            "recommendation": risk_info["recommendation"],
            "phishing_type": phishing_type,
            "reasons": reasons,
        }

    except Exception as e:
        return {
            "error": "Model inference failed",
            "details": str(e)
        }

