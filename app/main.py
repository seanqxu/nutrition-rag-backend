from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from app.config import get_settings
from app.rag_engine import RAGEngine, LabPanel
from app.document_ingestion import DocumentIngestion

# NEW IMPORTS
from app.modules.nutrition_mapper import NutritionMapper
from app.models.nutrition_targets import (
    NutritionTargetsRequest,
    NutritionTargetsResponse,
    UserProfile,
    NutritionTargets,
    MacroTarget,
    MicroTarget
)
from app.models.dietary_budgets import (
    BudgetRequest,
    BudgetResponse,
    DailyBudget,
    WeeklyBudget,
    MonthlyBudget
)

settings = get_settings()
MEDICAL_DISCLAIMER = (
    "IMPORTANT DISCLAIMER: This information is for educational purposes only "
    "and is not intended as medical advice, diagnosis, or treatment. Always "
    "consult with a qualified healthcare provider before making any changes "
    "to your diet, exercise, or medication regimen. The recommendations provided "
    "are based on general clinical guidelines and may not be appropriate for your "
    "specific health situation."
)


rag_engine: Optional[RAGEngine] = None
doc_ingestion: Optional[DocumentIngestion] = None
nutrition_mapper: Optional[NutritionMapper] = None  # NEW


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine, doc_ingestion, nutrition_mapper
    print("Initializing RAG Engine...")
    rag_engine = RAGEngine()
    doc_ingestion = DocumentIngestion()
    nutrition_mapper = NutritionMapper(rag_engine)  # NEW
    print("Nutrition RAG Backend ready.")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Nutrition RAG Backend",
    description="Evidence-based nutrition recommendations from lab panels",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LabPanelRequest(BaseModel):
    glucose_fasting: Optional[float] = Field(None, description="Fasting glucose in mg/dL")
    a1c: Optional[float] = Field(None, description="A1C percentage")
    total_cholesterol: Optional[float] = Field(None, description="Total cholesterol in mg/dL")
    ldl: Optional[float] = Field(None, description="LDL cholesterol in mg/dL")
    hdl: Optional[float] = Field(None, description="HDL cholesterol in mg/dL")
    triglycerides: Optional[float] = Field(None, description="Triglycerides in mg/dL")
    systolic_bp: Optional[float] = Field(None, description="Systolic blood pressure in mmHg")
    diastolic_bp: Optional[float] = Field(None, description="Diastolic blood pressure in mmHg")
    bmi: Optional[float] = Field(None, description="Body Mass Index")
    egfr: Optional[float] = Field(None, description="Estimated glomerular filtration rate")  # NEW


class SourceInfo(BaseModel):
    guideline: str
    source: str
    relevance_score: float


class RecommendationResponse(BaseModel):
    recommendation: str
    sources: list[SourceInfo]
    lab_panel: dict
    disclaimer: str = MEDICAL_DISCLAIMER


class HealthResponse(BaseModel):
    status: str
    ollama_status: str
    qdrant_status: str
    collection_stats: dict


@app.get("/health", response_model=HealthResponse)
async def health_check():
    import ollama as ollama_client
    from qdrant_client import QdrantClient
    
    try:
        ollama_client.list()
        ollama_status = "healthy"
    except Exception as e:
        ollama_status = f"unhealthy: {str(e)}"
    
    try:
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        client.get_collections()
        qdrant_status = "healthy"
    except Exception as e:
        qdrant_status = f"unhealthy: {str(e)}"
    
    try:
        stats = doc_ingestion.get_collection_stats()
    except:
        stats = {}
    
    overall = "healthy" if ollama_status == "healthy" and qdrant_status == "healthy" else "degraded"
    
    return HealthResponse(
        status=overall,
        ollama_status=ollama_status,
        qdrant_status=qdrant_status,
        collection_stats=stats
    )


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendation(lab_panel: LabPanelRequest):
    values = [v for v in lab_panel.model_dump().values() if v is not None]
    if not values:
        raise HTTPException(status_code=400, detail="At least one lab value must be provided")
    
    panel = LabPanel(
        glucose_fasting=lab_panel.glucose_fasting,
        a1c=lab_panel.a1c,
        total_cholesterol=lab_panel.total_cholesterol,
        ldl=lab_panel.ldl,
        hdl=lab_panel.hdl,
        triglycerides=lab_panel.triglycerides,
        systolic_bp=lab_panel.systolic_bp,
        diastolic_bp=lab_panel.diastolic_bp,
        bmi=lab_panel.bmi,
        egfr=lab_panel.egfr  # NEW
    )
    
    try:
        result = rag_engine.generate_recommendation(panel)
        return RecommendationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendation: {str(e)}")


# NEW ENDPOINT: Nutrition Targets
@app.post("/nutrition/targets", response_model=NutritionTargetsResponse)
async def generate_nutrition_targets(request: NutritionTargetsRequest):
    """
    Generate personalized nutrition targets from lab results.
    Uses existing RAGEngine to retrieve clinical guidelines.
    """
    
    # Create LabPanel from request
    lab_panel = LabPanel(
        glucose_fasting=request.glucose_fasting,
        a1c=request.a1c,
        total_cholesterol=request.total_cholesterol,
        ldl=request.ldl,
        hdl=request.hdl,
        triglycerides=request.triglycerides,
        systolic_bp=request.systolic_bp,
        diastolic_bp=request.diastolic_bp,
        bmi=request.bmi,
        egfr=request.egfr
    )
    
    try:
        # Use RAG to generate targets
        targets_dict = nutrition_mapper.generate_nutrition_targets(
            lab_panel=lab_panel,
            age=request.user_profile.age,
            sex=request.user_profile.sex,
            weight_kg=request.user_profile.weight_kg,
            height_cm=request.user_profile.height_cm,
            activity_level=request.user_profile.activity_level
        )
        
        # Build lab summary
        lab_summary = _build_lab_summary(lab_panel)
        
        # Convert to models
        nutrition_targets = NutritionTargets(
            carbohydrates=MacroTarget(**targets_dict["carbohydrates"]),
            protein=MacroTarget(**targets_dict["protein"]),
            fat=MacroTarget(**targets_dict["fat"]),
            saturated_fat=MacroTarget(**targets_dict["saturated_fat"]) if targets_dict.get("saturated_fat") else None,
            sodium=MicroTarget(**targets_dict["sodium"]),
            fiber=MicroTarget(**targets_dict["fiber"]) if targets_dict.get("fiber") else None,
            daily_calories=targets_dict["daily_calories"],
            safety_alerts=targets_dict["safety_alerts"],
            foods_to_emphasize=targets_dict["foods_to_emphasize"],
            foods_to_limit=targets_dict["foods_to_limit"],
            clinical_sources=targets_dict["clinical_sources"]
        )
        
        return NutritionTargetsResponse(
            lab_summary=lab_summary,
            nutrition_targets=nutrition_targets,
            lifestyle_recommendations=[
                "Aim for 150 minutes of moderate-intensity aerobic activity per week",
                "Resistance training 2-3 times per week",
                "Aim for 7-9 hours of sleep nightly"
            ],
            sources_used=targets_dict["clinical_sources"],
            disclaimer=MEDICAL_DISCLAIMER
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating nutrition targets: {str(e)}"
        )


# NEW ENDPOINT: Budget Calculator
@app.post("/budgets/calculate", response_model=BudgetResponse)
async def calculate_dietary_budget(
    request: BudgetRequest,
    user_profile: UserProfile,
    nutrition_targets: NutritionTargets
):
    """
    Calculate dietary budgets (daily, weekly, or monthly).
    """
    
    try:
        # Calculate TDEE
        tdee = _calculate_tdee(
            user_profile.age,
            user_profile.sex,
            user_profile.weight_kg,
            user_profile.height_cm,
            user_profile.activity_level
        )
        
        # Create daily budget
        daily = DailyBudget(
            calories=nutrition_targets.daily_calories,
            protein_g=nutrition_targets.protein.daily_target or 90,
            carbohydrates_g=nutrition_targets.carbohydrates.daily_target or 150,
            fat_g=nutrition_targets.fat.daily_target or 70,
            fiber_g=nutrition_targets.fiber.daily_limit if nutrition_targets.fiber else 30,
            sodium_mg=nutrition_targets.sodium.daily_limit
        )
        
        # Weekly budget
        weekly = None
        if request.time_period in ["weekly", "monthly"]:
            weekly = WeeklyBudget(
                total_calories=daily.calories * 7,
                total_protein_g=daily.protein_g * 7,
                total_carbohydrates_g=daily.carbohydrates_g * 7,
                total_fat_g=daily.fat_g * 7,
                total_fiber_g=daily.fiber_g * 7,
                total_sodium_mg=daily.sodium_mg * 7,
                daily_average=daily
            )
        
        # Monthly budget
        monthly = None
        if request.time_period == "monthly":
            from datetime import datetime
            days = 30
            
            monthly = MonthlyBudget(
                month=datetime.now().strftime("%B %Y"),
                days_in_month=days,
                total_calories=daily.calories * days,
                total_protein_g=daily.protein_g * days,
                total_carbohydrates_g=daily.carbohydrates_g * days,
                total_fat_g=daily.fat_g * days,
                total_fiber_g=daily.fiber_g * days,
                total_sodium_mg=daily.sodium_mg * days,
                daily_average=daily
            )
        
        notes = []
        if request.allow_flexibility:
            notes.append("Budget allows 10-15% daily flexibility while maintaining weekly/monthly targets")
        
        if nutrition_targets.safety_alerts:
            notes.extend(nutrition_targets.safety_alerts)
        
        return BudgetResponse(
            time_period=request.time_period,
            daily_budget=daily,
            weekly_budget=weekly,
            monthly_budget=monthly,
            calculation_method=f"Mifflin-St Jeor TDEE with {user_profile.activity_level} activity",
            tdee=tdee,
            notes=notes
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating budget: {str(e)}"
        )


@app.post("/ingest")
async def ingest_documents(background_tasks: BackgroundTasks):
    background_tasks.add_task(doc_ingestion.ingest_directory)
    return {"status": "ingestion started", "message": "Check /health for progress"}


@app.get("/stats")
async def get_stats():
    return doc_ingestion.get_collection_stats()


@app.get("/models")
async def list_models():
    import ollama as ollama_client
    try:
        models = ollama_client.list()
        return {
            "models": [m["name"] for m in models["models"]],
            "active_model": settings.ollama_model,
            "embedding_model": settings.ollama_embedding_model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


@app.get("/disclaimer")
async def get_disclaimer():
    return {"disclaimer": MEDICAL_DISCLAIMER.strip()}


# Helper functions
def _build_lab_summary(lab_panel) -> str:
    """Build human-readable lab summary"""
    
    parts = []
    
    if lab_panel.glucose_fasting or lab_panel.a1c:
        if lab_panel.a1c and lab_panel.a1c >= 6.5:
            parts.append(f"diabetes (A1C {lab_panel.a1c}%)")
        elif lab_panel.a1c and lab_panel.a1c >= 5.7:
            parts.append(f"prediabetes (A1C {lab_panel.a1c}%)")
        elif lab_panel.glucose_fasting and lab_panel.glucose_fasting >= 126:
            parts.append(f"elevated fasting glucose ({lab_panel.glucose_fasting} mg/dL)")
    
    if lab_panel.ldl and lab_panel.ldl >= 130:
        parts.append(f"elevated LDL cholesterol ({lab_panel.ldl} mg/dL)")
    
    if lab_panel.systolic_bp and lab_panel.systolic_bp >= 130:
        parts.append(f"elevated blood pressure ({lab_panel.systolic_bp}/{lab_panel.diastolic_bp} mmHg)")
    
    if lab_panel.egfr:
        if lab_panel.egfr < 60:
            parts.append(f"moderate kidney disease (eGFR {lab_panel.egfr})")
        elif lab_panel.egfr < 90:
            parts.append(f"mild kidney function decline (eGFR {lab_panel.egfr})")
    
    if not parts:
        return "Lab values are generally within normal ranges"
    
    return "Lab results indicate: " + ", ".join(parts)


def _calculate_tdee(age: int, sex: str, weight_kg: float, height_cm: float, activity_level: str) -> int:
    """Calculate TDEE using Mifflin-St Jeor"""
    
    if sex.lower() == "male":
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
    
    multipliers = {
        "sedentary": 1.2,
        "lightly_active": 1.375,
        "moderately_active": 1.55,
        "very_active": 1.725,
        "extra_active": 1.9
    }
    
    multiplier = multipliers.get(activity_level, 1.55)
    return int(bmr * multiplier)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port, reload=True)
