"""
Dietary Budgets Models
Weekly and monthly calorie/macro budgets
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date


class DailyBudget(BaseModel):
    """Daily nutritional budget"""
    
    calories: int = Field(..., ge=1200, le=4000)
    protein_g: float = Field(..., ge=30, le=300)
    carbohydrates_g: float = Field(..., ge=50, le=500)
    fat_g: float = Field(..., ge=20, le=200)
    fiber_g: float = Field(..., ge=20, le=80)
    sodium_mg: float = Field(..., ge=500, le=3000)


class WeeklyBudget(BaseModel):
    """Weekly nutritional budget"""
    
    total_calories: int
    total_protein_g: float
    total_carbohydrates_g: float
    total_fat_g: float
    total_fiber_g: float
    total_sodium_mg: float
    
    daily_average: DailyBudget


class MonthlyBudget(BaseModel):
    """Monthly nutritional budget"""
    
    month: str
    days_in_month: int = Field(..., ge=28, le=31)
    
    total_calories: int
    total_protein_g: float
    total_carbohydrates_g: float
    total_fat_g: float
    total_fiber_g: float
    total_sodium_mg: float
    
    daily_average: DailyBudget


class BudgetRequest(BaseModel):
    """Request to calculate dietary budget"""
    
    time_period: str = Field(..., pattern="^(daily|weekly|monthly)$")
    start_date: Optional[date] = None
    allow_flexibility: bool = Field(default=True)


class BudgetResponse(BaseModel):
    """Response with calculated budget"""
    
    time_period: str
    daily_budget: DailyBudget
    weekly_budget: Optional[WeeklyBudget] = None
    monthly_budget: Optional[MonthlyBudget] = None
    
    calculation_method: str
    tdee: int
    
    notes: List[str] = Field(default_factory=list)
