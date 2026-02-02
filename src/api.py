from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import analysis functions
from src.inference import (
    predict_email,
    get_risk_message,
    detect_links,
    detect_urgency_words,
    detect_phishing_type,
    load_model,
)


# Load model on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
    yield
    logger.info("Shutting down application")


# Initialize FastAPI app
app = FastAPI(
    title="Phishing Email Detection API",
    description="ML-powered API for detecting phishing emails",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your website domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request model
class AnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="Email text to analyze (10-10000 characters)"
    )


# Response model
class AnalyzeResponse(BaseModel):
    risk_score: float
    risk_category: str
    risk_level: str
    explanation: str
    recommendation: str
    phishing_type: str
    reasons: list[str]


# Health check endpoint
@app.get("/", tags=["Health"])
def health_check():
    return {
        "message": "Phishing Detection API is running",
    }


@app.get("/health", tags=["Health"])
def detailed_health():
    try:
        from src.inference import model
        model_loaded = model is not None
    except:
        model_loaded = False

    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "api_version": "1.0.0"
    }


# Main analysis endpoint
@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
def analyze_email(request: AnalyzeRequest):
    try:
        text = request.text.strip()

        # Validate minimum word count
        word_count = len(text.split())
        if word_count < 5:
            raise HTTPException(
                status_code=400,
                detail="Email text too short. Minimum 5 words required."
            )

        # Get prediction
        score = predict_email(text)
        risk_info = get_risk_message(score)

        # Detect features
        links = detect_links(text)
        urgent_words = detect_urgency_words(text)

        # Build reasons list
        reasons = []
        if links:
            reasons.append(f"Suspicious link detected ({len(links)} link(s) found)")
        else:
            reasons.append("No suspicious links found")

        if urgent_words:
            reasons.append(
                f"Urgent or pressure language detected ({', '.join(urgent_words[:5])})"
            )
        else:
            reasons.append("No urgent or threatening language detected")

        # Determine phishing type
        phishing_type = "legitimate"
        if score >= 0.40:
            phishing_type = detect_phishing_type(text, links, urgent_words)

        # Log the analysis
        logger.info(f"Analyzed email: risk_score={score:.4f}, category={risk_info['level']}")

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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Endpoint not found",
        "detail": "The requested endpoint does not exist"
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "error": "Internal server error",
        "detail": "An unexpected error occurred"
    }
