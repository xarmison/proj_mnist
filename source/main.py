from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
import re, io, base64
from PIL import Image
import numpy as np
import joblib
import wandb

artifact_model_name = 'proj_mnist/model_export:latest'

run = wandb.init(project='proj_mnist', job_type='api')

# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.post('/predict/')
async def predict(drawing_data: str) -> dict:
    """
        Receive the base64 encoding of an image 
        containing a handwritten digit and make 
        predictions using an ml model.
        
        Args:
            drawing_data (str): Base64 encoding of an image.

        Returns:
            dict: JSON response containing the predictions
    """
    # Convert data in url to numpy array
    img_str = re.search(r'base64,(.*)', drawing_data.replace(' ', '+')).group(1)
    img_bytes = io.BytesIO(base64.b64decode(img_str))
    img = Image.open(img_bytes)
    
    # Normalize pixel values
    input = np.array(img)[:, :, 0:1].reshape((1, 28*28)) / 255.0

    model_export_path = run.use_artifact(artifact_model_name).file()
    clf = joblib.load(model_export_path)

    predictions = clf.predict_proba(input)[0]
    
    return { 
        'result': 1,
        'error': '',
        'data': list(predictions)
    }

app.mount('/', StaticFiles(directory='./source/static', html = True), name='static')
