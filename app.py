from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import uvicorn

from Text_summariser.pipeline.prediction import PredictionPipeline

app = FastAPI(title="Text Summarizer",
        description="Text Summarization using Google's FLAN-T5",
        version="1.0.0"
)

templates = Jinja2Templates(directory="templates")
predictor = PredictionPipeline()

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
    )

@app.post("/predict")
async def predict_route(request: Request, text: str = Form(...)):
    try:
        summary = predictor.predict(text)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "summary": summary,
                "original": text
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "error": str(e)
            }
        )

@app.post("/api/summarize")
async def summarize(text: str):
    return {"summary": predictor.predict(text)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)