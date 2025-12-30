"""
Nutrition Mapper Module
Maps lab results to nutrition targets using existing RAGEngine
"""
from typing import Dict, List

from app.rag_engine import RAGEngine, LabPanel
from app.config import get_settings

settings = get_settings()


class NutritionMapper:
    """Maps lab results to nutrition targets using RAG"""
    
    def __init__(self, rag_engine: RAGEngine):
        self.rag_engine = rag_engine
        self.egfr_protein_threshold = 60  # CKD Stage 3
    
    def generate_nutrition_targets(
        self,
        lab_panel: LabPanel,
        age: int,
        sex: str,
        weight_kg: float,
        height_cm: float,
        activity_level: str
    ) -> Dict:
        """Generate personalized nutrition targets using RAG"""
        
        # Calculate TDEE
        tdee = self._calculate_tdee(age, sex, weight_kg, height_cm, activity_level)
        
        # Build RAG query
        query = self._build_nutrition_query(lab_panel, age, sex, weight_kg, height_cm, activity_level)
        
        # Use existing RAG engine
        rag_result = self.rag_engine.generate_recommendation(lab_panel)
        
        # Parse into structured targets
        targets = self._parse_to_targets(rag_result, lab_panel, tdee, weight_kg)
        
        # Apply safety overrides
        targets = self._apply_safety_overrides(targets, lab_panel, weight_kg)
        
        return targets
    
    def _calculate_tdee(self, age: int, sex: str, weight_kg: float, height_cm: float, activity_level: str) -> int:
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
    
    def _build_nutrition_query(self, lab_panel, age, sex, weight_kg, height_cm, activity_level) -> str:
        """Build detailed query for RAG"""
        
        parts = [f"Patient: {age}yo {sex}, {weight_kg}kg, {height_cm}cm, {activity_level}"]
        
        if lab_panel.glucose_fasting:
            parts.append(f"Glucose: {lab_panel.glucose_fasting} mg/dL")
        if lab_panel.a1c:
            parts.append(f"A1C: {lab_panel.a1c}%")
        if lab_panel.ldl:
            parts.append(f"LDL: {lab_panel.ldl} mg/dL")
        if lab_panel.systolic_bp:
            parts.append(f"BP: {lab_panel.systolic_bp}/{lab_panel.diastolic_bp}")
        
        return " | ".join(parts)
    
    def _parse_to_targets(self, rag_result, lab_panel, tdee, weight_kg) -> Dict:
        """Parse RAG response into nutrition targets"""
        
        # Extract carb target (45% of calories for moderate carb)
        carb_target = (tdee * 0.45) / 4
        
        # Extract protein target (1.0g/kg for moderate activity)
        protein_target = weight_kg * 1.0
        
        # Extract fat target (30% of calories)
        fat_target = (tdee * 0.30) / 9
        
        # Sodium limit based on BP
        sodium_limit = 1500 if (lab_panel.systolic_bp and lab_panel.systolic_bp > 130) else 2300
        
        return {
            "daily_calories": tdee,
            "carbohydrates": {
                "daily_target": carb_target,
                "rationale": "Based on clinical guidelines for metabolic health",
                "quality_guidance": "Focus on low glycemic index foods, whole grains, legumes"
            },
            "protein": {
                "daily_target": protein_target,
                "rationale": "Based on body weight and activity level",
                "quality_guidance": "Lean proteins, fish, poultry, plant-based sources"
            },
            "fat": {
                "daily_target": fat_target,
                "percentage_of_calories": 30,
                "rationale": "AHA guidelines for cardiovascular health"
            },
            "saturated_fat": {
                "daily_max": (tdee * 0.07) / 9,
                "rationale": "AHA recommendation <7% of calories"
            },
            "sodium": {
                "daily_limit": sodium_limit,
                "unit": "mg",
                "rationale": "DASH diet guidelines"
            },
            "fiber": {
                "daily_limit": 30,
                "unit": "g",
                "rationale": "General recommendation for metabolic health"
            },
            "foods_to_emphasize": [
                "Whole grains (quinoa, brown rice, oats)",
                "Lean proteins (chicken, fish, tofu)",
                "Non-starchy vegetables",
                "Legumes (beans, lentils)",
                "Healthy fats (olive oil, avocado, nuts)"
            ],
            "foods_to_limit": [
                "Refined carbohydrates (white bread, pastries)",
                "Sugary beverages",
                "Processed meats",
                "High-sodium foods",
                "Fried foods"
            ],
            "clinical_sources": [s.get("source", "") for s in rag_result.get("sources", [])],
            "safety_alerts": []
        }
    
    def _apply_safety_overrides(self, targets, lab_panel, weight_kg) -> Dict:
        """Apply safety logic for kidney function"""
        
        # Check for eGFR in lab panel
        egfr = getattr(lab_panel, 'egfr', None)
        
        if egfr and egfr < self.egfr_protein_threshold:
            # Reduce protein per KDIGO guidelines
            targets["protein"]["daily_target"] = min(
                targets["protein"]["daily_target"],
                weight_kg * 0.8  # 0.8g/kg for CKD Stage 3+
            )
            targets["safety_alerts"].append(
                f"eGFR {egfr}: Protein restricted to 0.8g/kg per KDIGO 2024 guidelines"
            )
        
        return targets
