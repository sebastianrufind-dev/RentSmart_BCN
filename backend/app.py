from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ============================================================
# DATASET
# ============================================================

rent_ranges = {
    "08001": {1:(1100,1400),2:(1500,1900),3:(1900,2300)},
    "08007": {1:(1200,1500),2:(1600,2100),3:(2100,2600)},
    "08014": {1:(900,1200),2:(1200,1500),3:(1500,1900)},
    "08028": {1:(1000,1300),2:(1350,1750),3:(1800,2300)},
    "08017": {1:(1200,1600),2:(1700,2200),3:(2300,2900)},
    "08012": {1:(1000,1400),2:(1400,1800),3:(1850,2350)},
}

neigh_list = list(rent_ranges.keys())
bed_list = [1, 2, 3]
apartments_per_combo = 50

np.random.seed(42)
data = {"Apartment_name": [], "Rental_price": [], "number_bedrooms": [], "neigbourhood": []}
apt_id = 1
for n in neigh_list:
    for b in bed_list:
        for _ in range(apartments_per_combo):
            pmin, pmax = rent_ranges[n][b]
            price = np.random.randint(pmin, pmax + 1)
            data["Apartment_name"].append(f"Apartment {apt_id}")
            data["Rental_price"].append(price)
            data["number_bedrooms"].append(b)
            data["neigbourhood"].append(n)
            apt_id += 1

df_apartments = pd.DataFrame(data)

# ============================================================
# RECOMMENDER LOGIC (same as your notebook, but without widgets)
# ============================================================

def generate_candidates(df, budget, bedrooms, neighbourhood=None, tol=0.10):
    min_p, max_p = budget*(1-tol), budget*(1+tol)
    c = df[(df.number_bedrooms >= bedrooms) &
           (df.Rental_price >= min_p) &
           (df.Rental_price <= max_p)].copy()
    if neighbourhood:
        sub = c[c.neigbourhood == neighbourhood]
        if not sub.empty:
            c = sub
    return c

def score_candidates(candidates, budget, bedrooms, neighbourhood=None,
                     wp=0.6, wb=0.25, wn=0.15):
    c = candidates.copy()
    tol_price = budget * 1.10

    def price_score(p):
        return 1.0 if p <= budget else max(0, 1 - (p-budget)/(tol_price-budget))

    c["price_score"] = c.Rental_price.apply(price_score)

    def bedroom_score_fn(b):
        if b == bedrooms:
            return 1.0
        elif b > bedrooms:
            return 0.8
        else:
            return 0.0

    c["bedroom_score"] = c.number_bedrooms.apply(bedroom_score_fn)

    c["neigbourhood_score"] = np.where(
        (c.neigbourhood == neighbourhood) | (neighbourhood is None),
        1.0,
        0.5
    )

    c["total_score"] = (
        wp * c.price_score +
        wb * c.bedroom_score +
        wn * c.neigbourhood_score
    )

    return c

def rerank(c, top=10):
    return c.sort_values(["total_score", "Rental_price"],
                         ascending=[False, True]).head(top)

def recommendation_pipeline(df, budget, bedrooms, neigh, top=10, tol=0.10):
    cand = generate_candidates(df, budget, bedrooms, neigh, tol)
    if cand.empty:
        return cand
    scored = score_candidates(cand, budget, bedrooms, neigh)
    return rerank(scored, top)

# ============================================================
# REGRESSION MODEL FOR LANDLORDS
# ============================================================

X = pd.get_dummies(df_apartments[["number_bedrooms", "neigbourhood"]], drop_first=True)
y = df_apartments["Rental_price"]

reg_model = LinearRegression()
reg_model.fit(X, y)

def predict_rent_price(bedrooms, neighbourhood_zip):
    feat = pd.DataFrame({
        "number_bedrooms": [bedrooms],
        "neigbourhood": [neighbourhood_zip]
    })
    feat_enc = pd.get_dummies(feat, columns=["neigbourhood"])
    feat_enc = feat_enc.reindex(columns=X.columns, fill_value=0)
    pred = reg_model.predict(feat_enc)[0]
    return float(pred)

# ============================================================
# FASTAPI APP + CORS
# ============================================================

app = FastAPI(title="Barcelona Rental Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # open for demo; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Pydantic models ---------

class TenantRequest(BaseModel):
    budget: float
    bedrooms: int
    neighbourhood_zip: Optional[str] = None
    top_n: int = 5

class TenantRecommendation(BaseModel):
    apartment_name: str
    rental_price: float
    number_bedrooms: int
    neighbourhood: str
    total_score: float

class TenantResponse(BaseModel):
    recommendations: list[TenantRecommendation]
    probability: float

class LandlordRequest(BaseModel):
    bedrooms: int
    neighbourhood_zip: str

class LandlordResponse(BaseModel):
    recommended_price: float
    avg_price: float
    min_price: float
    max_price: float

# --------- Endpoints ---------

@app.post("/tenant_recommendations", response_model=TenantResponse)
def tenant_recommendations(req: TenantRequest):
    rec = recommendation_pipeline(
        df_apartments,
        budget=req.budget,
        bedrooms=req.bedrooms,
        neigh=req.neighbourhood_zip,
        top= req.top_n
    )
    if rec.empty:
        return TenantResponse(recommendations=[], probability=0.0)

    candidates = generate_candidates(
        df_apartments,
        budget=req.budget,
        bedrooms=req.bedrooms,
        neighbourhood=req.neighbourhood_zip,
    )
    matches = len(candidates)
    total = len(df_apartments) if req.neighbourhood_zip is None else \
            len(df_apartments[df_apartments.neigbourhood == req.neighbourhood_zip])
    probability = matches / total if total > 0 else 0.0

    recs_out = [
        TenantRecommendation(
            apartment_name=row.Apartment_name,
            rental_price=float(row.Rental_price),
            number_bedrooms=int(row.number_bedrooms),
            neighbourhood=row.neigbourhood,
            total_score=float(row.total_score),
        )
        for _, row in rec.iterrows()
    ]

    return TenantResponse(recommendations=recs_out, probability=probability)

@app.post("/landlord_recommendation", response_model=LandlordResponse)
def landlord_recommendation(req: LandlordRequest):
    price = predict_rent_price(req.bedrooms, req.neighbourhood_zip)
    comps = df_apartments[
        (df_apartments.number_bedrooms == req.bedrooms) &
        (df_apartments.neigbourhood == req.neighbourhood_zip)
    ]
    if comps.empty:
        avg_price = min_price = max_price = price
    else:
        avg_price = float(comps.Rental_price.mean())
        min_price = float(comps.Rental_price.min())
        max_price = float(comps.Rental_price.max())

    return LandlordResponse(
        recommended_price=price,
        avg_price=avg_price,
        min_price=min_price,
        max_price=max_price,
    )
