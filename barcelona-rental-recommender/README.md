# Barcelona Rental Recommender

This project contains a FastAPI backend that simulates a rental market in Barcelona
and exposes two main endpoints:

- `/tenant_recommendations`: given a budget, number of bedrooms and (optional) neighbourhood ZIP code,
  returns a ranked list of apartments and the probability of finding a match.
- `/landlord_recommendation`: given number of bedrooms and neighbourhood ZIP,
  returns a recommended rental price using a linear regression model trained on the simulated data.

## Structure

- `backend/app.py` — FastAPI application with data simulation, recommendation logic, and endpoints.
- `backend/requirements.txt` — Python dependencies for the backend.

## Running locally

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

You can then import this repository into tools like Lovable to build a frontend
with a Tenant/Landlord toggle that calls these endpoints.
