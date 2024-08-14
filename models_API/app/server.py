import os
import requests
import joblib
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.utils import SwisstopoTileFetcher, ConvertRGBToFeedModel, create_grayscale_tiles
from toloboy.toloboy import from_LAB_to_RGB_img
import base64
from io import BytesIO
from PIL import Image

app = FastAPI()

# Setup Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

# URLs to the models on GitHub Releases
model_urls = {
    "base_model": "https://github.com/tololojo/Colorizing_API/releases/download/v0.1/base_model.joblib",
    "HyperUnet_retrain_augmented_noise_corrected_Adam": "https://github.com/tololojo/Colorizing_API/releases/download/v0.1/HyperUnet_retrain_augmented_noise_corrected_Adam.joblib",
    "HyperUnet_retrain_no_augmented_corrected_Adam": "https://github.com/tololojo/Colorizing_API/releases/download/v0.1/HyperUnet_retrain_no_augmented_corrected_Adam.joblib"
}

# Local paths where models will be stored
model_paths = {
    "base_model": "app/base_model.joblib",
    "HyperUnet_retrain_augmented_noise_corrected_Adam": "app/HyperUnet_retrain_augmented_noise_corrected_Adam.joblib",
    "HyperUnet_retrain_no_augmented_corrected_Adam": "app/HyperUnet_retrain_no_augmented_corrected_Adam.joblib"
}

def download_model(url, destination):
    if not os.path.exists(destination):
        print(f"Downloading {destination}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(destination, 'wb') as f:
                f.write(response.content)
            print(f"Model saved as {destination}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download model from {url}. Error: {e}")
    else:
        print(f"{destination} already exists, skipping download.")

# Download models
for model_name, url in model_urls.items():
    download_model(url, model_paths[model_name])

# Load models into a dictionary
models = {name: joblib.load(path) for name, path in model_paths.items()}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Renders the main page with the form for user input.
    """
    return templates.TemplateResponse("index.html", {"request": request, "model_names": model_paths.keys()})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    coordinates: str = Form(...),
    zoom_level: int = Form(...),
    layer_name: str = Form(default="ch.swisstopo.swissimage"),
    times: str = Form(default="current"),
    model_name: str = Form(...),
):
    """
    Handles the prediction request from the web form, fetches the tiles,
    processes them, and returns the original and colorized images.
    """

    if model_name not in models:
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, 
             "error": "Invalid model name", 
             "model_names": model_paths.keys()
             })

    model = models[model_name]

    # Convert form data into required formats
    coordinates = [tuple(map(float, coord.split(','))) for coord in coordinates.split(';')]
    times = times.split(',')

    # Fetch the tiles
    tile_fetcher = SwisstopoTileFetcher(coordinates, zoom_level, layer_name, times)
    tile_fetcher.fetch_all_tiles()
    fetched_tiles = tile_fetcher.get_tiles()

    # Convert fetched tiles to grayscale and prepare for the model
    grayscale_tiles = create_grayscale_tiles(fetched_tiles)

    predictions = {}

    for (longitude, latitude), image_list in grayscale_tiles.items():
        predictions[(longitude, latitude)] = []
        for time, image, image_color, orig_L, origi_AB in image_list:
            # Run the model prediction
            predicted_AB = model.predict(orig_L.reshape(1, *orig_L.shape), verbose=0)
            predicted_image = from_LAB_to_RGB_img(L=orig_L, AB=predicted_AB)
            predicted_image_pil = Image.fromarray(predicted_image, mode="RGB")

            # Convert images to base64
            original_image_base64 = image_to_base64(image_color)
            original_image_grey_base64 = image_to_base64(image)
            predicted_image_base64 = image_to_base64(predicted_image_pil)

            # Store the results
            predictions[(longitude, latitude)].append({
                "time": time,
                "original_color_image": original_image_base64,
                "original_grey_image": original_image_grey_base64,
                "predicted_image": predicted_image_base64,
            })

    return templates.TemplateResponse("index.html", {
        "request": request, 
        "predictions": predictions, 
        "model_names": model_paths.keys()
        })

def image_to_base64(image: Image.Image) -> str:
    """
    Converts a PIL image to a base64 encoded string.
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
