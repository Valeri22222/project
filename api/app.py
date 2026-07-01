from fastapi import FastAPI
from pydantic import BaseModel
from model_service import service
import uvicorn

app = FastAPI(title="СберАвтоподписка API")

class VisitData(BaseModel):
    total_events: float
    unique_pages: float
    catalog_visits: float
    order_visits: float
    visit_number: float
    visit_hour: float
    visit_weekday: float
    visit_month: float
    is_weekend: int
    is_paid: int
    has_catalog_view: int
    has_order_view: int

@app.on_event("startup")
async def startup():
    service.load()

@app.post("/predict")
async def predict(data: VisitData):
    return service.predict(data.dict())

@app.get("/health")
async def health():
    return {"status": "ok", "loaded": service.loaded}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)