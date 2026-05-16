from fastapi import FastAPI
from starlette.responses import Response, RedirectResponse
import uvicorn
import os

from Text_summariser.pipeline.prediction import PredictionPipeline


app = FastAPI()


@app.get("/", tags=["authentication"])
async def home():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        os.system("python main.py")

        return Response(
            content="Training Successful !!",
            media_type="text/plain"
        )

    except Exception as e:
        return Response(
            content=f"Error Occurred! {e}",
            media_type="text/plain"
        )


@app.post("/predict")
async def predict_route(text: str):

    try:
        obj = PredictionPipeline()

        summary = obj.predict(text)

        return {
            "summary": summary
        }

    except Exception as e:
        return Response(
            content=f"Prediction Error: {e}",
            media_type="text/plain"
        )


if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080
    )