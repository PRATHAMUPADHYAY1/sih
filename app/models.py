# app/models.py
from pydantic import BaseModel


class PromotionPlan(BaseModel):
    scheme_name: str
    plan: str


class PredictionRequest(BaseModel):
    post_office_name: str
    top_n_schemes: int = 3
    include_neighbor_vote: bool = False


class PlanRequest(BaseModel):
    post_office_name: str
    top_n_schemes: int = 3
    include_neighbor_vote: bool = True


class TrendsRequest(BaseModel):
    district_name: str