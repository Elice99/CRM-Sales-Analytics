"""
================================================================================
MAIN.PY - FASTAPI PRODUCTION ENDPOINT FOR SALES PIPELINE MODEL
================================================================================

PURPOSE:
    This script builds a production-ready API using FastAPI that:
    1. Loads the trained model from model.py
    2. Exposes HTTP endpoints for making predictions
    3. Accepts deals data via API request 
    4. Handles all feature engineering internally on the backend.
    5. Returns predictions (WIN vs LOST) with probabilities
    6. Includes error handling and validation

================================================================================
"""

# ============================================================================
# IMPORTS - Required libraries
# ============================================================================

from fastapi import FastAPI, HTTPException  # FastAPI framework for building APIs
from fastapi.responses import JSONResponse # Custom JSON responses for errors
from contextlib import asynccontextmanager # For managing startup/shutdown events in FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Handle Cross-Origin requests
from pydantic import BaseModel, Field  # Data validation using Python type hints
import os  # For paths and environment variables
import pickle  # For loading the saved model file
import pandas as pd  
import numpy as np  
import logging  # For logging events to console/files
from typing import List, Optional  
from datetime import datetime  
import uvicorn  # ASGI server to run FastAPI
import traceback  # For detailed error messages
from features import compute_all_features, ensure_features_exist  # Shared feature engineering

# ============================================================================
# CONFIGURE LOGGING
# ============================================================================
"""
Logging: Records important events in a file or console
This helps debug issues in production
"""

logging.basicConfig(
    level=logging.INFO,  # Show INFO level and above (WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_logs.txt'),  # Save logs to a file
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)  # Create a logger for this module

# ============================================================================
# FASTAPI APP + CORS  (created ONCE, after the lifespan handler below)
# ============================================================================
"""
FastAPI is a modern web framework for building APIs in Python.
NOTE: Earlier versions of this file created the app TWICE. The second
`app = FastAPI(...)` silently replaced the first, which dropped the CORS
middleware and docs configuration. Keep exactly one app instance.
"""

# ============================================================================
# LOAD MODEL (Global - loads once when API starts)
# ============================================================================
"""
Global variables store data that persists across all API requests
We load the model once at startup, not for each request (better performance)
"""

# Resolve the model path relative to THIS file, not the current working
# directory. (Running uvicorn from another folder used to break model loading.)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sales_pipeline_model.pkl')
artifacts = None  # This will store the loaded model, but as of now it is set to None until the model is actually loaded.


def load_model():
    """
    FUNCTION: load_model()
    
    PURPOSE: Load the pickled model file when the API starts
    
    RETURNS: dict with model artifacts or None if failed
    
    PROCESS:
        1. Try to open the pickle file
        2. Deserialize the model
        3. Log success
        4. If error, log and return None
    """
    global artifacts
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            artifacts = pickle.load(f) #deserializes (loads) your trained model Stores it into the global artifacts variable
        logger.info(" Model loaded successfully")
        return artifacts
    except FileNotFoundError:
        logger.error(f"Model file not found: {MODEL_PATH}")
        return None  # If the file path is wrong or missing, Logs a clear error to aviod crashing the api
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None  #If anything unexpected happens (corrupt file, bad pickle, etc.), Logs “Error loading model” with details


# ============================================================================
# PYDANTIC MODELS - Data validation schemas
# ============================================================================
"""
Pydantic models define the structure of data that the API accepts/returns
They automatically validate input and return helpful error messages if invalid

Think of them as contracts: "Your request must have this data with these types"
"""


class DealPredictionInput(BaseModel):
    """
    INPUT MODEL: Data required for a single deal prediction
    
    This defines what data a user must send to the /predict endpoint
    Pydantic will validate that all fields have the correct type
    
    EXAMPLE JSON INPUT:
    {
        "sales_agent": "john_smith",
        "product": "gtx_basic",
        "account": "acme_corporation",
        "sector": "finance",
        "office_location": "united_states",
        "manager": "jane_doe",
        "regional_office": "central",
        "company_size": "large",
        "year_established": 1995,

    }
    """
    
    # REQUIRED FIELDS (no default value)
    sales_agent: str = Field(..., description="Name of the sales agent")
    product: str = Field(..., description="Product name")
    account: str = Field(..., description="Customer account name")
    sector: str = Field(..., description="Industry sector")
    office_location: str = Field(..., description="Office location")
    engage_date: str = Field(..., description="Deal engagement date (YYYY-MM-DD format)")
    
    # OPTIONAL FIELDS (with defaults to fill any missing data)
    manager: Optional[str] = Field(default="unknown", description="Manager name")
    regional_office: Optional[str] = Field(default="central", description="Regional office")
    year_established: Optional[int] = Field(default=1995, description="Year company was established")
    employees: Optional[float] = Field(default=8.0, description="number of employees")
    


class BatchPredictionInput(BaseModel):
    """
    INPUT MODEL: Multiple deals for batch prediction
    
    EXAMPLE JSON INPUT:
    {
        "deals": [
            {deal fields...},
            {deal fields...}
        ]
    }
    """
    deals: List[DealPredictionInput] = Field(..., description="List of deals to predict")


class PredictionOutput(BaseModel):
    """
    OUTPUT MODEL: Single prediction result
    
    This is what the API returns after making a prediction
    
    EXAMPLE JSON OUTPUT:
    {
        "predicted_outcome": "WON",
        "probability_won": 0.72,
        "probability_lost": 0.28,
        "confidence": 0.72,
        "timestamp": "2024-01-15T10:30:45.123456"
    }
    """
    predicted_outcome: str = Field(..., description="Predicted outcome: 'WON' or 'LOST'")
    probability_won: float = Field(..., description="Probability of winning (0-1)")
    probability_lost: float = Field(..., description="Probability of losing (0-1)")
    confidence: float = Field(..., description="Model confidence in prediction (0-1)")
    timestamp: str = Field(..., description="Timestamp of prediction")


class BatchPredictionOutput(BaseModel):
    """
    OUTPUT MODEL: Multiple predictions
    """
    predictions: List[PredictionOutput] = Field(..., description="List of predictions")
    total_deals: int = Field(..., description="Total deals processed")
    deals_won: int = Field(..., description="Deals predicted to win")
    deals_lost: int = Field(..., description="Deals predicted to lose")
    summary: str = Field(..., description="Summary of results")


class ModelMetrics(BaseModel):
    """
    OUTPUT MODEL: Model performance metrics
    """
    model_type: str = Field(..., description="Type of model (e.g., Pipeline)")
    test_roc_auc: float = Field(..., description="Test set ROC-AUC score")
    total_features: int = Field(..., description="Number of features")
    categorical_features: int = Field(..., description="Number of categorical features")
    numerical_features: int = Field(..., description="Number of numerical features")
    status: str = Field(..., description="Model status (loaded/failed)")


class HealthCheck(BaseModel):
    """
    OUTPUT MODEL: Health check response
    """
    status: str = Field(..., description="API status: 'healthy' or 'unhealthy'")
    model_loaded: bool = Field(..., description="Is model loaded?")
    timestamp: str = Field(..., description="Current timestamp")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
"""
All feature engineering lives in features.py (shared with test.py) so the
API and the batch scorer always compute features exactly the same way.
"""



def make_prediction(request_data):
    """
    FUNCTION: make_prediction(request_data)
    
    PURPOSE: Generate a single prediction from the model
    
    PARAMETERS:
        request_data: DealPredictionInput object with deal information
    
    RETURNS:
        dict: Contains predicted outcome, probabilities, and confidence
    
    PROCESS:
        1. Convert request data to DataFrame
        2. Ensure all features exist
        3. Extract features in correct order
        4. Get model probability prediction
        5. Apply threshold to get binary prediction
        6. Calculate confidence
        7. Format and return results
    """
    try:
        # Convert request data to dictionary then to DataFrame
        deal_dict = request_data.model_dump()  # Convert Pydantic model to dict (Pydantic v2)
        df = pd.DataFrame([deal_dict])  # Convert to single-row DataFrame
        
        #  CRITICAL: Compute all engineered features from the raw data
        df = compute_all_features(df, artifacts)
        
        # Ensure all features exist
        df = ensure_features_exist( df, artifacts['numerical_features'], artifacts['categorical_features'])
        
        # Extract features in correct order
        feature_columns = artifacts['feature_names']
        missing_cols = [c for c in feature_columns if c not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        X = df[feature_columns]
        
        # Get probability prediction from model
        probabilities = artifacts['model'].predict_proba(X)[:, 1]
        prob_won = float(probabilities[0])  # Extract single probability value
        prob_lost = 1 - prob_won
        
        # Apply threshold (0.5) for binary prediction
        threshold = 0.5
        if prob_won >= threshold:
            predicted_outcome = "WON"
        else:
            predicted_outcome = "LOST"
        
        # Calculate confidence (how sure is the model?)
        confidence = max(prob_won, prob_lost)
        
        # Format and return results
        return {
            "predicted_outcome": predicted_outcome,
            "probability_won": round(prob_won, 4),
            "probability_lost": round(prob_lost, 4),
            "confidence": round(confidence, 4),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        # If prediction fails, log error and re-raise
        logger.error(f"Error making prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


def make_batch_predictions(deals):
    """
    FUNCTION: make_batch_predictions(deals)

    PURPOSE: Predict a whole batch in one pass (much faster than calling
             make_prediction per deal: one DataFrame, one predict_proba call).
    """
    deal_dicts = [d.model_dump() for d in deals]
    df = pd.DataFrame(deal_dicts)

    df = compute_all_features(df, artifacts)
    df = ensure_features_exist(df, artifacts['numerical_features'], artifacts['categorical_features'])
    X = df[artifacts['feature_names']]

    probabilities = artifacts['model'].predict_proba(X)[:, 1]

    results = []
    for prob_won in probabilities:
        prob_won = float(prob_won)
        prob_lost = 1.0 - prob_won
        results.append({
            "predicted_outcome": "WON" if prob_won >= 0.5 else "LOST",
            "probability_won": round(prob_won, 4),
            "probability_lost": round(prob_lost, 4),
            "confidence": round(max(prob_won, prob_lost), 4),
            "timestamp": datetime.now().isoformat()
        })
    return results


# ============================================================================
# API ENDPOINTS
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):

    global artifacts

    logger.info("=" * 80)
    logger.info("SALES PIPELINE API - STARTING UP")
    logger.info("=" * 80)

    artifacts = load_model()

    if artifacts is None:
        logger.error("Failed to load model")
    else:
        logger.info(f"Model loaded with {len(artifacts['feature_names'])} features")

    yield

    logger.info("Shutting down API")


app = FastAPI(
    title="Sales Pipeline Prediction API",
    description="Predicts sales deal outcomes (WON vs LOST) using raw data input.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Demo only - replace with your frontend domain in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    # NOTE: don't combine allow_credentials=True with wildcard origins -
    # browsers reject that combination.
)


@app.get("/")
async def root():
    """
    ENDPOINT: GET /
    
    PURPOSE: Welcome message and basic info
    
    RETURNS: Welcome message
    """
    return {
        "message": "Welcome to Sales Pipeline Prediction API",
        "documentation": "Visit /docs for interactive documentation",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "metrics": "GET /metrics",
            "single_prediction": "POST /predict",
            "batch_prediction": "POST /predict-batch"
        }
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """
    ENDPOINT: GET /health
    
    PURPOSE: Check if API is running and model is loaded
    
    RETURNS: Health status
    
    USE CASE: Use this endpoint to monitor if the API is alive
             Can be used by load balancers to know if server is healthy
    """
    model_loaded = artifacts is not None
    
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metrics", response_model=ModelMetrics)
async def get_metrics():
    """
    ENDPOINT: GET /metrics
    
    PURPOSE: Get model performance metrics
    
    RETURNS: Model information and performance scores
    
    USE CASE: Understand model capabilities and performance
    """
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(artifacts['model']).__name__,
        "test_roc_auc": artifacts['test_roc_auc'],
        "total_features": len(artifacts['feature_names']),
        "categorical_features": len(artifacts['categorical_features']),
        "numerical_features": len(artifacts['numerical_features']),
        "status": "loaded"
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict_single(deal: DealPredictionInput):
    """
    ENDPOINT: POST /predict
    
    PURPOSE: Predict outcome for a single deal
    
    PARAMETERS:
        deal (DealPredictionInput): The deal data (provided in JSON body)
    
    RETURNS:
        PredictionOutput: Prediction result with probabilities and confidence
    
    EXAMPLE REQUEST (curl):
        curl -X POST "http://127.0.0.1:9696/predict" \
             -H "Content-Type: application/json" \
             -d '{
                    "sales_agent": "john_smith",
                    "product": "gtx_basic",
                    "account": "acme_corp",
                    "sector": "finance",
                    "office_location": "united_states",
                    "engage_date": "2026-01-15"
                }'
    
    EXAMPLE RESPONSE:
        {
            "predicted_outcome": "LOST",
            "probability_won": 0.3856,
            "probability_lost": 0.6144,
            "confidence": 0.6144,
            "timestamp": "2024-01-15T10:30:45.123456"
        }
    
    ERROR HANDLING:
        - 500: If model not loaded or prediction fails
        - 422: If request data validation fails
    """
    logger.info(f"Received prediction request for deal: {deal.account}")
    
    # Check if model is loaded
    if artifacts is None:
        logger.error("Model not loaded!")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Make prediction
    result = make_prediction(deal)
    
    logger.info(f"Prediction result: {result['predicted_outcome']} ({result['confidence']:.2%} confidence)")
    
    return result


@app.post("/predict-batch", response_model=BatchPredictionOutput)
async def predict_batch(batch: BatchPredictionInput):
    """
    ENDPOINT: POST /predict-batch
    
    PURPOSE: Predict outcomes for multiple deals at once
    
    PARAMETERS:
        batch (BatchPredictionInput): List of deals to predict
    
    RETURNS:
        BatchPredictionOutput: List of predictions plus summary statistics
    
    EXAMPLE REQUEST (curl):
        curl -X POST "http://127.0.0.1:9696/predict-batch" \
             -H "Content-Type: application/json" \
             -d '{
                 "deals": [
                     {deal 1 data...},
                     {deal 2 data...},
                     {deal 3 data...}
                 ]
             }'
    
    EXAMPLE RESPONSE:
        {
            "predictions": [
                {prediction 1...},
                {prediction 2...},
                {prediction 3...}
            ],
            "total_deals": 3,
            "deals_won": 1,
            "deals_lost": 2,
            "summary": "Analyzed 3 deals: 1 (33.3%) predicted to WIN, 2 (66.7%) predicted to LOST"
        }
    
    WHY BATCH ENDPOINT?
        - More efficient than multiple single requests
        - Useful for processing many deals at once
        - Reduces overhead and improves performance
    """
    logger.info(f"Received batch prediction request for {len(batch.deals)} deals")
    
    # Check if model is loaded
    if artifacts is None:
        logger.error("Model not loaded!")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Predict the whole batch in one vectorized pass
    try:
        predictions = make_batch_predictions(batch.deals)
    except Exception as e:
        logger.error(f"Error predicting batch: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")
    
    # Calculate summary statistics
    won_count = sum(1 for p in predictions if p['predicted_outcome'] == 'WON')
    lost_count = sum(1 for p in predictions if p['predicted_outcome'] == 'LOST')
    total = len(predictions)
    
    won_pct = (won_count / total * 100) if total > 0 else 0
    lost_pct = (lost_count / total * 100) if total > 0 else 0
    
    summary = f"Analyzed {total} deals: {won_count} ({won_pct:.1f}%) predicted to WIN, {lost_count} ({lost_pct:.1f}%) predicted to LOST"
    
    logger.info(summary)
    
    return {
        "predictions": predictions,
        "total_deals": total,
        "deals_won": won_count,
        "deals_lost": lost_count,
        "summary": summary
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):

    """
    FUNCTION: http_exception_handler
    
    PURPOSE: Custom handler for HTTP errors
    
    Logs all HTTP errors for debugging
    """
    
    logger.error(f"HTTP Error: {exc.status_code} - {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):

    """
    FUNCTION: general_exception_handler
    
    PURPOSE: Catch-all handler for unexpected errors
    
    Logs all unexpected errors for debugging
    """
    
    logger.error(f"Unexpected error: {str(exc)}")
    logger.error(traceback.format_exc())

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )



# ============================================================================
# RUN THE API
# ============================================================================
# NOTE: This block MUST sit at module level (zero indentation). It was
# previously indented, so Python treated it as extra code inside
# general_exception_handler() above - meaning `python main.py` silently
# exited without ever starting the server.

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STARTING SALES PIPELINE API")
    print("=" * 80)
    print("\nAccess API Documentation:")
    print("  - Swagger UI: http://localhost:9696/docs")
    print("  - ReDoc:      http://localhost:9696/redoc")
    print("  - Health:     http://localhost:9696/health")
    print("  - Metrics:    http://localhost:9696/metrics")
    print("\n" + "=" * 80 + "\n")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=9696,
    )

"""
================================================================================
DEPLOYMENT INSTRUCTIONS
================================================================================

1. DEVELOPMENT (Local Testing):
   python main.py
   Then visit: http://localhost:9696/docs

2. PRODUCTION DEPLOYMENT:
   
   Option A - Using Gunicorn (Linux/Mac):
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:9696 main:app
   
   Option B - Using Docker:
   Create Dockerfile:
   ----
   FROM python:3.11
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["python", "main.py"]
   ----
   
   Build and run:
   docker build -t sales-api .
   docker run -p 9696:9696 sales-api

   Option C - Cloud Deployment:
   - AWS: Use Elastic Beanstalk or EC2
   - Google Cloud: Use Cloud Run
   - Azure: Use App Service
   - Heroku: git push heroku main (after setup)

3. SECURITY CONSIDERATIONS:
   - Add authentication (JWT tokens)
   - Add rate limiting
   - Use HTTPS (SSL/TLS certificates)
   - Validate all inputs strictly
   - Don't expose sensitive model info in responses
   - Use environment variables for configuration

4. MONITORING:
   - Log all predictions
   - Track API response times
   - Monitor model performance on new data
   - Set up alerts for errors
   - Compare predictions vs actual outcomes

5. UPDATING THE MODEL:
   - Retrain model.py periodically
   - Save updated sales_pipeline_model.pkl
   - Restart the API
   - No code changes needed!

================================================================================
"""