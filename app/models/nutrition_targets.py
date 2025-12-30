"""
Nutrition Targets Models
Based on lab results and clinical guidelines
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class MacroTarget(BaseModel):
    """Target for a single macronutrient"""
    
    daily_min: Optional[float] = Field(None, description="Minimum grams per day")
    daily_max: Optional[float] = Field(None, description="Maximum grams per day")
    daily_target: Optional[float] = Field(None, description="Target grams per day")
    percentage_of_calories: Optional[float] = Field(None, ge=0, le=100)
    rationale: str = Field(..., description="Clinical reasoning from guidelines")
    quality_guidance: Optional[str] = Field(None, description="Quality/source recommendations")


class MicroTarget(BaseModel):
    """Target for micronutrients (sodium, fiber, etc.)"""
    
    daily_limit: Optional[float] = Field(None, description="Daily limit/target")
    unit: str = Field(..., description="Unit (mg, g)")
    rationale: str = Field(..., description="Clinical reasoning")


class NutritionTargets(BaseModel):
    """Complete nutrition targets based on lab results"""
    
    # Macronutrients
    carbohydrates: MacroTarget
    protein: MacroTarget
    fat: MacroTarget
    
    # Fat breakdown
    saturated_fat: Optional[MacroTarget] = None
    unsaturated_fat: Optional[MacroTarget] = None
    
    # Micronutrients
    sodium: MicroTarget
    fiber: Optional[MicroTarget] = None
    potassium: Optional[MicroTarget] = None
    
    # Calories
    daily_calories: int = Field(..., ge=1200, le=4000)
    
    # Special considerations
    safety_alerts: List[str] = Field(
        default_factory=list,
        description="Critical safety considerations (e.g., kidney function)"
    )
    
    foods_to_emphasize: List[str] = Field(default_factory=list)
    foods_to_limit: List[str] = Field(default_factory=list)
    
    clinical_sources: List[str] = Field(
        default_factory=list,
        description="Guidelines used (ADA, AHA, DASH, KDIGO)"
    )


class UserProfile(BaseModel):
    """User demographics for TDEE calculation"""
    
    age: int = Field(..., ge=18, le=120)
    sex: str = Field(..., pattern="^(male|female)$")
    weight_kg: float = Field(..., gt=30, lt=300)
    height_cm: float = Field(..., gt=100, lt=250)
    activity_level: str = Field(
        ...,
        pattern="^(sedentary|lightly_active|moderately_active|very_active|extra_active)$"
    )


class NutritionTargetsRequest(BaseModel):
    """Request to generate nutrition targets"""
    
    # Lab panel
    glucose_fasting: Optional[float] = None
    a1c: Optional[float] = None
    total_cholesterol: Optional[float] = None
    ldl: Optional[float] = None
    hdl: Optional[float] = None
    triglycerides: Optional[float] = None
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    bmi: Optional[float] = None
    egfr: Optional[float] = Field(None, description="Kidney function")
    
    # User profile
    user_profile: UserProfile


class NutritionTargetsResponse(BaseModel):
    """Response with nutrition targets"""
    
    lab_summary: str
    nutrition_targets: NutritionTargets
    lifestyle_recommendations: List[str] = Field(default_factory=list)
    sources_used: List[str] = Field(default_factory=list)
    disclaimer: str
