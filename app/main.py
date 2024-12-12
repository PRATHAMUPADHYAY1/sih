# app/main.py
from fastapi import FastAPI, HTTPException
from app.models import PredictionRequest, PlanRequest, TrendsRequest
from app.data_loading import (
    x_df,
    x_df_ins,
    final_df,
    model_1,
    model_2,
    model_1_ins,
    model_2_ins,
    numeric_cols_1,
    numeric_cols_2,
    numeric_cols_1_ins,
    numeric_cols_2_ins,
    district_data,
)
from app.projections import calculate_projections
from app.utils import collate_predictions, get_demographics
from app.promotion_plan import collate_and_generate_plan

app = FastAPI(title="Post Office Scheme Recommendation API")


@app.get("/")
def read_root():
    return {"message": "Welcome to the Post Office Scheme Recommendation API"}


@app.post("/plot_district_trends")
def plot_district_trends(request: TrendsRequest):
    """
    Endpoint to return 5-year workforce projections for a given district.
    """
    district_name = request.district_name
    growth_rates = {
        "Workforce_Participation": 0.02,
        "Workforce_Growth": 0.05,
        "Elderly_Workers": 0.03,
        "Urbanization": 0.02,
        "Female_Literacy": 0.04,
    }

    try:
        projections = calculate_projections(district_name, district_data, growth_rates)
        return {
            "district_name": district_name,
            "projections": {
                "Years": [f"Year +{i}" for i in range(5)],
                "Workforce_Participation": projections["Workforce_Participation"],
                "Projected_Workforce": projections["Projected_Workforce"],
                "Elderly_Workers": projections["Elderly_Workers"],
                "Urban_Workforce": projections["Urban_Workforce"],
                "Female_Workforce_Inclusion": projections["Female_Workforce_Inclusion"],
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        raise HTTPException(
            status_code=500, detail="An error occurred while calculating projections."
        )


@app.post("/predict_schemes")
def predict_schemes(request: PredictionRequest):
    post_office_name = request.post_office_name
    top_n_schemes = request.top_n_schemes
    include_neighbor_vote = request.include_neighbor_vote

    if post_office_name not in final_df["Post Office Name"].values:
        raise HTTPException(status_code=404, detail="Post Office not found.")

    try:
        schemes = collate_predictions(
            post_office_name,
            model_1,
            model_2,
            x_df,
            final_df,
            numeric_cols_1,
            numeric_cols_2,
            months=23,
            month_offset=1,
            top_n_schemes=2,
            include_neighbor_vote=include_neighbor_vote,
        )
        insurances = collate_predictions(
            post_office_name,
            model_1_ins,
            model_2_ins,
            x_df_ins,
            final_df,
            numeric_cols_1_ins,
            numeric_cols_2_ins,
            months=23,
            month_offset=1,
            top_n_schemes=1,
            include_neighbor_vote=include_neighbor_vote,
            is_insurance=True,
        )
        return {
            "post_office_name": post_office_name,
            "recommended_schemes": schemes,
            "recommended_insurances": insurances,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/promotion_plans")
def get_promotion_plans(request: PlanRequest):
    post_office_name = request.post_office_name
    top_n_schemes = request.top_n_schemes
    include_neighbor_vote = request.include_neighbor_vote

    if post_office_name not in final_df["Post Office Name"].values:
        raise HTTPException(status_code=404, detail="Post Office not found.")

    try:
        plans = collate_and_generate_plan(
            post_office_name,
            model_1,
            model_2,
            x_df,
            final_df,
            numeric_cols_1,
            numeric_cols_2,
            months=23,
            month_offset=1,
            top_n_schemes=top_n_schemes,
            include_neighbor_vote=include_neighbor_vote,
        )
        return {
            "post_office_name": post_office_name,
            "promotion_plans": [p.dict() for p in plans],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/demographics/{post_office_name}")
def get_po_demographics(post_office_name: str):
    if post_office_name not in final_df["Post Office Name"].values:
        raise HTTPException(status_code=404, detail="Post Office not found.")
    data = get_demographics(post_office_name)
    return data.to_dict(orient="records")