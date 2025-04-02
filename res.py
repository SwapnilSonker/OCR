import time
import easyocr
import spacy
import re
from dateutil import parser
import cv2
from fastapi import FastAPI, UploadFile, File
import numpy as np

app = FastAPI()

# Load OCR and NLP models
reader = easyocr.Reader(['en'])
nlp = spacy.load("en_core_web_trf")

def process_image(image_bytes):
    # Convert image bytes to NumPy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Perform OCR
    ocr_result = reader.readtext(threshold_image)
    
    # Combine OCR result into a single text block
    text = " ".join([result[1] for result in ocr_result])
    
    # Apply NLP (NER) to extract named entities
    doc = nlp(text)
    entities = {"venue": "", "date": "", "time": "", "topic": "", "phone": "", "mode": ""}
    
    # Extract topic (block letters or italics or larger text)
    topic_candidates = [result[1] for result in ocr_result if result[1].isupper() or "*" in result[1]]
    if topic_candidates:
        entities["topic"] = " ".join(topic_candidates)
    
    # Extract venue and mode of event
    venue_keywords = ["conference hall", "auditorium", "room", "venue", "hall"]
    address_found = False
    
    for ent in doc.ents:
        if ent.label_ in ["GPE", "ORG", "FAC"] or any(kw in ent.text.lower() for kw in venue_keywords):
            entities["venue"] = ent.text
            address_found = True
        elif ent.label_ == "DATE":
            entities["date"] = ent.text
        elif ent.label_ == "TIME":
            entities["time"] = ent.text
    
    # Use regex for phone number
    phone_pattern = r"\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
    entities["phone"] = re.findall(phone_pattern, text)
    
    # Determine mode of event
    if "hybrid" in text.lower():
        entities["mode"] = "hybrid"
    elif not address_found:
        entities["mode"] = "online"
    else:
        entities["mode"] = "offline"
    
    # Extract date and time
    date_string = re.search(r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4})\b", text)
    if date_string:
        entities["date"] = parser.parse(date_string.group())
    
    return entities

@app.post("/extract")
async def extract_info(file: UploadFile = File(...)):
    start_time = time.time()
    image_bytes = await file.read()
    result = process_image(image_bytes)
    end_time = time.time()
    result["processing_time"] = f"{end_time - start_time:.4f} seconds"
    return result
