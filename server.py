from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from inference import predict
import os
app = FastAPI()

templates = Jinja2Templates(directory="templates")
UPLOAD_FOLDER = 'imgs/'


@app.get("/", response_class=HTMLResponse)
def write_home():
    return templates.TemplateResponse("index.html")


@app.post("/")
async def handle_form(request: Request, file: UploadFile = File(...)):
    file_name = "img2inference.jpg"
    file_location = os.path.join(
        UPLOAD_FOLDER,
        file_name
    )
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    prediction, proba = predict()
    return templates.TemplateResponse("index.html", {"request": request,
                                                     "image_loc": file_location,
                                                     "prediction": prediction,
                                                     "proba": proba})

