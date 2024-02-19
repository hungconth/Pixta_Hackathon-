from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
from datetime import datetime
from AI import process_image 
from fastapi import UploadFile, File, HTTPException
import cv2
import numpy as np
import shutil
import os
from datetime import datetime
import uvicorn
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InferenceAttributes(BaseModel):
    race: str
    age: str
    emotion: str
    gender: str
    skintone: str
    masked: str

# A simple class to represent a clothing item
class ClothingItem(BaseModel):
    name: str
    size: Optional[str] = None
    color: Optional[str] = None
    material: Optional[str] = None


# A placeholder function for actual recommendation logic
def get_clothing_recommendations(attributes: InferenceAttributes) -> List[ClothingItem]:
    recommendations = []

    # Example logic: recommending based on age and gender
    if attributes.age.startswith("20") and attributes.gender == "Male":
        recommendations.append(ClothingItem(name="Casual Shirt", size="M", color="Blue"))
    if attributes.age.startswith("30") and attributes.gender == "Female":
        recommendations.append(ClothingItem(name="Elegant Dress", size="M", color="Red"))

    # Add more complex recommendation logic based on other attributes
    # ...

    return recommendations




@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/inference/")
# async def inference(image: bytes):
#     # Here you will handle the image and call your API inference
#     # Example: response = await call_api_inference(image)
#     # return response
#     pass

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
from datetime import datetime
import json  # Import json

@app.post("/inference/")
async def inference(image: UploadFile = File(...)):
    # Folder where images will be saved
    folder = "static\saved_images"
    os.makedirs(folder, exist_ok=True)

    try:
        # Construct a file path with a timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = f"{folder}/{timestamp}_{image.filename}"

        # Read image in chunks and save as a temporary file
        with open(file_path, 'wb') as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # AI
        result = process_image(file_path)
        print("result", result)
        print(type(result))

        # Parse the result string to a Python object
        result_data = json.loads(result)
        print(type(result_data))
        # Return the data as a JSON response
        return JSONResponse(content=result_data)

    except HTTPException as e:
        # If there's an HTTP error, re-raise it
        raise e
    except Exception as e:
        # For any other exceptions, return an error message
        return JSONResponse(content={"error": str(e)}, status_code=500)



@app.post("/recommend/", response_model=List[ClothingItem])
async def recommend(attributes: InferenceAttributes):
    try:
        # Get clothing recommendations based on the attributes
        clothing_recommendations = get_clothing_recommendations(attributes)

        # Return the clothing recommendations
        return clothing_recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api-result/", response_model=List[ClothingItem])
async def api_result(image: UploadFile = File(...)):
    # Folder where images will be saved
    folder = "saved_images"
    os.makedirs(folder, exist_ok=True)

    try:
        # Construct a file path with a timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = f"{folder}/{timestamp}_{image.filename}"

        # Read image in chunks and save as a temporary file
        with open(file_path, 'wb') as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # AI Inference
        inference_result = process_image(file_path)
        attributes = InferenceAttributes(**inference_result)

        # Recommendation based on inference
        clothing_recommendations = get_clothing_recommendations(attributes)

        # Assuming we only want to recommend shirts and pants, we filter the results
        shirt_and_pants_recommendations = [
            item for item in clothing_recommendations if item.name in ["Shirt", "Pants"]
        ]
        print("shirt_and_pants_recommendations", shirt_and_pants_recommendations)

        return shirt_and_pants_recommendations
    except HTTPException as e:
        # If there's an HTTP error, re-raise it
        raise e
    except Exception as e:
        # For any other exceptions, return an error message
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)

