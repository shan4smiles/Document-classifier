# remember to pip install altair in version 4
import io
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch.nn.functional as F
from easyocr import Reader
from PIL import Image
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3ForSequenceClassification, LayoutLMv3Processor, LayoutLMv3TokenizerFast 

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MICROSOFT_MODEL_NAME = "microsoft/layoutlmv3-base"
MODEL_NAME = "curiousily/layoutlmv3-financial-document-classification"

def preprocess_image(image: Image) -> Image:
    # Convert image to RGB format if it's not already
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

def create_bounding_box(bbox_data, width_scale: float, height_scale: float):
    xs = []
    ys = []
    for x, y in bbox_data:
        xs.append(x)
        ys.append(y)
 
    left = int(min(xs) * width_scale)
    top = int(min(ys) * height_scale)
    right = int(max(xs) * width_scale)
    bottom = int(max(ys) * height_scale)
 
    return [left, top, right, bottom]

@st.experimental_singleton
def create_ocr_reader():
    return Reader(["en"])

@st.experimental_singleton
def create_processor():
    feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False) 
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(MICROSOFT_MODEL_NAME) 
    return LayoutLMv3Processor (feature_extractor, tokenizer)

@st.experimental_singleton
def create_model():
    model = LayoutLMv3ForSequenceClassification.from_pretrained(MODEL_NAME)
    return model.eval().to(DEVICE)

def predict(image_input, reader: Reader, processor: LayoutLMv3Processor, model: LayoutLMv3ForSequenceClassification):
    # Check if the input is a file path (string)
    if isinstance(image_input, str):
        image = Image.open(image_input)
    # Check if the input is a PIL Image
    elif isinstance(image_input, Image.Image):
        image = image_input
    # Check if the input is image bytes
    elif isinstance(image_input, bytes):
        bytes_data = io.BytesIO(image_input)
        image = Image.open(bytes_data)
    else:
        raise ValueError("Invalid input type. Supported types: str (file path), PIL.Image, bytes")
    
    ocr_result = reader.readtext(image)

    width, height = image.size
    width_scale = 1000 / width
    height_scale = 1000 / height
    
    words = []
    boxes = []

    for bbox, word, confidence in ocr_result:
        words.append(word)
        boxes.append(create_bounding_box(bbox, width_scale, height_scale))
    
    encoding = processor(image, words, boxes=boxes, max_length=512, padding= "max_length", truncation = True, return_tensors="pt")

    with torch.inference_mode():
        output = model(input_ids = encoding["input_ids"].to(DEVICE), attention_mask = encoding["attention_mask"].to(DEVICE), bbox = encoding["bbox"].to(DEVICE), pixel_values = encoding["pixel_values"].to(DEVICE))

        
    logits =  output.logits
    predicted_class = logits.argmax()
    probabilities = F.softmax(logits, dim=-1).flatten().tolist()

    return predicted_class.detach().item(), probabilities

reader = create_ocr_reader()
processor = create_processor()
model = create_model()

uploaded_file = st.file_uploader("Upload Document Image", ["jpg","png"])

if uploaded_file is not None:
    uploaded_file_type = uploaded_file.name.split(".")[-1].lower()
    
    # Check if the uploaded file is in a supported format (jpg or png)
    if uploaded_file_type in ["jpg", "jpeg", "png"]:
        image_input = uploaded_file.read()
        predicted_class, probabilities = predict(image_input, reader, processor, model)
        
        # Display the image
        image = Image.open(io.BytesIO(image_input))
        image = preprocess_image(image)
        st.image(image, "Your Document")
        
        predicted_labels = model.config.id2label[predicted_class]
        st.markdown(f"Predicted label: **{predicted_labels}**")
    else:
        st.error("Invalid file format. Please upload a JPG or PNG file.")
    