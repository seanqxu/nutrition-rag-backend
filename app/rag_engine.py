from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

settings = get_settings()


@dataclass
class LabPanel:
    glucose_fasting: Optional[float] = None
    a1c: Optional[float] = None
    total_cholesterol: Optional[float] = None
    ldl: Optional[float] = None
    hdl: Optional[float] = None
    triglycerides: Optional[float] = None
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    bmi: Optional[float] = None
    egfr: Optional[float] = None


@dataclass
class RetrievedContext:
    content: str
    source: str
    guideline_type: str
    score: float


class RAGEngine:
    def __init__(self):
        self.qdrant = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
    
    def _get_embedding(self, text: str) -> List[float]:
        response = ollama.embeddings(
            model=settings.ollama_embedding_model,
            prompt=text
        )
        return response["embedding"]

    def _build_lab_query(self, lab_panel: LabPanel) -> str:
        conditions = []
        
        if lab_panel.glucose_fasting:
            if lab_panel.glucose_fasting >= 126:
                conditions.append("diabetic blood glucose management diet")
            elif lab_panel.glucose_fasting >= 100:
                conditions.append("prediabetes blood sugar control nutrition")
        
        if lab_panel.a1c:
            if lab_panel.a1c >= 6.5:
                conditions.append("diabetes A1C dietary management")
            elif lab_panel.a1c >= 5.7:
                conditions.append("prediabetes A1C prevention diet")
        
        if lab_panel.total_cholesterol and lab_panel.total_cholesterol >= 200:
            conditions.append("high cholesterol heart healthy diet")
        
        if lab_panel.ldl and lab_panel.ldl >= 130:
            conditions.append("LDL cholesterol reduction dietary guidelines")
        
        if lab_panel.hdl and lab_panel.hdl < 40:
            conditions.append("increase HDL cholesterol nutrition")
        
        if lab_panel.triglycerides and lab_panel.triglycerides >= 150:
            conditions.append("lower triglycerides diet recommendations")
        
        if lab_panel.systolic_bp and lab_panel.diastolic_bp:
            if lab_panel.systolic_bp >= 140 or lab_panel.diastolic_bp >= 90:
                conditions.append("hypertension DASH diet blood pressure")
            elif lab_panel.systolic_bp >= 120 or lab_panel.diastolic_bp >= 80:
                conditions.append("elevated blood pressure dietary changes")
        
        if lab_panel.bmi:
            if lab_panel.bmi >= 30:
                conditions.append("obesity weight management nutrition plan")
            elif lab_panel.bmi >= 25:
                conditions.append("overweight healthy weight loss diet")
        
        return " ".join(conditions) if conditions else "general healthy eating guidelines"

    def _identify_relevant_guidelines(self, lab_panel: LabPanel) -> List[str]:
        guidelines = []
        
        if lab_panel.glucose_fasting or lab_panel.a1c:
            guidelines.append("ADA")
        
        if (lab_panel.total_cholesterol or lab_panel.ldl or 
            lab_panel.hdl or lab_panel.triglycerides):
            guidelines.extend(["AHA", "LIPID"])
        
        if lab_panel.systolic_bp or lab_panel.diastolic_bp:
            guidelines.extend(["DASH", "AHA"])
        
        return list(set(guidelines)) if guidelines else ["GENERAL"]
    
    def retrieve_context(self, lab_panel: LabPanel, top_k: int = None) -> List[RetrievedContext]:
        query = self._build_lab_query(lab_panel)
        query_embedding = self._get_embedding(query)
        
        relevant_guidelines = self._identify_relevant_guidelines(lab_panel)
        
        results = self.qdrant.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_embedding,
            limit=top_k or settings.top_k_results,
            query_filter=Filter(
                should=[
                    FieldCondition(
                        key="guideline_type",
                        match=MatchValue(value=g)
                    )
                    for g in relevant_guidelines
                ]
            ) if relevant_guidelines else None
        )
        
        return [
            RetrievedContext(
                content=hit.payload["content"],
                source=hit.payload["source"],
                guideline_type=hit.payload["guideline_type"],
                score=hit.score
            )
            for hit in results
        ]

    def _build_prompt(self, lab_panel: LabPanel, contexts: List[RetrievedContext]) -> str:
        lab_summary = []
        if lab_panel.glucose_fasting:
            lab_summary.append(f"Fasting Glucose: {lab_panel.glucose_fasting} mg/dL")
        if lab_panel.a1c:
            lab_summary.append(f"A1C: {lab_panel.a1c}%")
        if lab_panel.total_cholesterol:
            lab_summary.append(f"Total Cholesterol: {lab_panel.total_cholesterol} mg/dL")
        if lab_panel.ldl:
            lab_summary.append(f"LDL: {lab_panel.ldl} mg/dL")
        if lab_panel.hdl:
            lab_summary.append(f"HDL: {lab_panel.hdl} mg/dL")
        if lab_panel.triglycerides:
            lab_summary.append(f"Triglycerides: {lab_panel.triglycerides} mg/dL")
        if lab_panel.systolic_bp and lab_panel.diastolic_bp:
            lab_summary.append(f"Blood Pressure: {lab_panel.systolic_bp}/{lab_panel.diastolic_bp} mmHg")
        if lab_panel.bmi:
            lab_summary.append(f"BMI: {lab_panel.bmi}")
        
        context_text = "\n\n---\n\n".join([
            f"[{ctx.guideline_type}] {ctx.content}"
            for ctx in contexts
        ])
        
        lab_str = chr(10).join(lab_summary)
        
        prompt = f"""You are a nutrition advisor providing evidence-based dietary recommendations.

Based on the following lab results and clinical guidelines, provide personalized nutrition recommendations.

## Patient Lab Results
{lab_str}

## Clinical Guidelines Reference
{context_text}

## Instructions
1. Analyze the lab results and identify areas of concern
2. Provide specific, actionable nutrition recommendations based on the clinical guidelines
3. Include daily targets for key nutrients where applicable
4. Suggest specific foods to include and limit
5. Be encouraging but honest about the importance of dietary changes

Provide your recommendations in a clear, organized format."""

        return prompt

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _call_llm(self, prompt: str) -> str:
        response = ollama.chat(
            model=settings.ollama_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a knowledgeable nutrition advisor who provides evidence-based dietary recommendations grounded in clinical guidelines from AHA, ADA, and DASH. You are not a doctor and always recommend consulting healthcare providers for medical decisions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.7,
                "num_predict": 1024
            }
        )
        return response["message"]["content"]
    
    def generate_recommendation(self, lab_panel: LabPanel) -> Dict[str, Any]:
        contexts = self.retrieve_context(lab_panel)
        prompt = self._build_prompt(lab_panel, contexts)
        recommendation = self._call_llm(prompt)
        
        return {
            "recommendation": recommendation,
            "sources": [
                {
                    "guideline": ctx.guideline_type,
                    "source": ctx.source,
                    "relevance_score": ctx.score
                }
                for ctx in contexts
            ],
            "lab_panel": {
                k: v for k, v in lab_panel.__dict__.items() if v is not None
            }
        }
