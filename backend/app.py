from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

app = FastAPI(title="Barcelona Rental Recommender API")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Barcelona Rental Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now (simple fix)
    allow_credentials=True,
    allow_methods=["*"],  # Allow POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)


# ----------------------------
# 1. Simulated dataset
# ----------------------------

rent_ranges = {
    "08001": {1: (1100, 1400), 2: (1500, 1900), 3: (1900, 2300)},
    "08007": {1: (1200, 1500), 2: (1600, 2100), 3: (2100, 2600)},
    "08014": {1: (900, 1200),  2: (1200, 1500), 3: (1500, 1900)},
    "08028": {1: (1000, 1300), 2: (1350, 1750), 3: (1800, 2300)},
    "08017": {1: (1200, 1600), 2: (1700, 2200), 3: (2300, 2900)},
    "08012": {1: (1000, 1400), 2: (1400, 1800), 3: (1850, 2350)},
}

neigh_list = list(rent_ranges.keys())
bed_list = [1, 2, 3]
apartments_per_combo = 50

np.random.seed(42)
data = {
    "Apartment_name": [],
    "Rental_price": [],
    "number_bedrooms": [],
    "neigbourhood": [],
}
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

# ----------------------------
# 2. Recommendation logic
# ----------------------------

def generate_candidates(df, budget: float, bedrooms: int, neighbourhood: Optional[str] = None, tol: float = 0.10):
    min_p, max_p = budget * (1 - tol), budget * (1 + tol)
    candidates = df[
        (df["number_bedrooms"] >= bedrooms)
        & (df["Rental_price"] >= min_p)
        & (df["Rental_price"] <= max_p)
    ].copy()
    if neighbourhood:
        subset = candidates[candidates["neigbourhood"] == neighbourhood]
        if not subset.empty:
            candidates = subset
    return candidates


def score_candidates(
    candidates: pd.DataFrame,
    budget: float,
    bedrooms: int,
    neighbourhood: Optional[str] = None,
    w_price: float = 0.6,
    w_bedrooms: float = 0.25,
    w_neighbourhood: float = 0.15,
):
    scored = candidates.copy()
    tol_price = budget * 1.10

    def price_score(p: float) -> float:
        if p <= budget:
            return 1.0
        return max(0.0, 1.0 - (p - budget) / (tol_price - budget))

    scored["price_score"] = scored["Rental_price"].apply(price_score)

    def bedroom_score_fn(b: int) -> float:
        if b == bedrooms:
            return 1.0
        if b > bedrooms:
            return 0.8
        return 0.0

    scored["bedroom_score"] = scored["number_bedrooms"].apply(bedroom_score_fn)

    scored["neigbourhood_score"] = np.where(
        (scored["neigbourhood"] == neighbourhood) | (neighbourhood is None),
        1.0,
        0.5,
    )

    scored["total_score"] = (
        w_price * scored["price_score"]
        + w_bedrooms * scored["bedroom_score"]
        + w_neighbourhood * scored["neigbourhood_score"]
    )

    return scored


def rerank(scored: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    return scored.sort_values(
        by=["total_score", "Rental_price"], ascending=[False, True]
    ).head(top_n)


def recommendation_pipeline(
    df: pd.DataFrame,
    budget: float,
    bedrooms: int,
    neighbourhood: Optional[str],
    top_n: int = 10,
    tol: float = 0.10,
) -> pd.DataFrame:
    candidates = generate_candidates(df, budget, bedrooms, neighbourhood, tol)
    if candidates.empty:
        return candidates
    scored = score_candidates(candidates, budget, bedrooms, neighbourhood)
    return rerank(scored, top_n)


# ----------------------------
# 3. Regression model (landlord)
# ----------------------------

X = pd.get_dummies(df_apartments[["number_bedrooms", "neigbourhood"]], drop_first=True)
y = df_apartments["Rental_price"]

reg_model = LinearRegression()
reg_model.fit(X, y)


def predict_rent_price(bedrooms: int, neighbourhood_zip: str) -> float:
    feat = pd.DataFrame(
        {"number_bedrooms": [bedrooms], "neigbourhood": [neighbourhood_zip]}
    )
    feat_enc = pd.get_dummies(feat, columns=["neigbourhood"])
    feat_enc = feat_enc.reindex(columns=X.columns, fill_value=0)
    pred = reg_model.predict(feat_enc)[0]
    return float(pred)


# ----------------------------
# 4. API schemas
# ----------------------------

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
    recommendations: List[TenantRecommendation]
    probability: float


class LandlordRequest(BaseModel):
    bedrooms: int
    neighbourhood_zip: str


class LandlordResponse(BaseModel):
    recommended_price: float
    avg_price: float
    min_price: float
    max_price: float


# ----------------------------
# 5. Endpoints
# ----------------------------

@app.post("/tenant_recommendations", response_model=TenantResponse)
def tenant_recommendations(req: TenantRequest):
    rec = recommendation_pipeline(
        df_apartments,
        budget=req.budget,
        bedrooms=req.bedrooms,
        neighbourhood=req.neighbourhood_zip,
        top_n=req.top_n,
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
    total = (
        len(df_apartments)
        if req.neighbourhood_zip is None
        else len(df_apartments[df_apartments["neigbourhood"] == req.neighbourhood_zip])
    )
    probability = matches / total if total > 0 else 0.0

    recs_out: List[TenantRecommendation] = []
    for _, row in rec.iterrows():
        recs_out.append(
            TenantRecommendation(
                apartment_name=row["Apartment_name"],
                rental_price=float(row["Rental_price"]),
                number_bedrooms=int(row["number_bedrooms"]),
                neighbourhood=row["neigbourhood"],
                total_score=float(row["total_score"]),
            )
        )

    return TenantResponse(recommendations=recs_out, probability=probability)


@app.post("/landlord_recommendation", response_model=LandlordResponse)
def landlord_recommendation(req: LandlordRequest):
    price = predict_rent_price(req.bedrooms, req.neighbourhood_zip)

    comps = df_apartments[
        (df_apartments["number_bedrooms"] == req.bedrooms)
        & (df_apartments["neigbourhood"] == req.neighbourhood_zip)
    ]
    if comps.empty:
        avg_price = min_price = max_price = price
    else:
        avg_price = float(comps["Rental_price"].mean())
        min_price = float(comps["Rental_price"].min())
        max_price = float(comps["Rental_price"].max())

    return LandlordResponse(
        recommended_price=price,
        avg_price=avg_price,
        min_price=min_price,
        max_price=max_price,
    )
