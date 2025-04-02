import time
import easyocr
import spacy
import re
from dateutil import parser
import cv2
from fastapi import FastAPI, UploadFile, File
import numpy as np
from easyocr import Reader

app = FastAPI()

# Load OCR and NLP models
# reader = easyocr.Reader(['en'])
# nlp = spacy.load('en_core_web_trf')

reader = Reader(['en'], gpu=True if cv2.cuda.getCudaEnabledDeviceCount() > 0 else False)
nlp = spacy.load('en_core_web_trf')  # Use transformer model for better accuracy


def process_image(image_bytes):
    """
    Process an image with enhanced pre-processing and entity extraction
    """
    # Initialize models with better configuration
    # reader = Reader(['en'], gpu=True if cv2.cuda.getCudaEnabledDeviceCount() > 0 else False)
    # nlp = spacy.load('en_core_web_trf')  # Use transformer model for better accuracy
    
    # Convert image bytes to NumPy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Create multiple processed versions of the image for better OCR coverage
    processed_images = []
    
    # Original image converted to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_images.append(("gray", gray))
    
    # Version 1: High contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    processed_images.append(("clahe", enhanced))
    
    # Version 2: Denoised image 
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    processed_images.append(("denoised", denoised))
    
    # Version 3: Sharpened image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    processed_images.append(("sharpened", sharpened))
    
    # Version 4: Binarized with Otsu
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(("binary_otsu", binary_otsu))
    
    # Version 5: Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    processed_images.append(("adaptive", adaptive))
    
    # Version 6: Dilated to connect broken text
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(binary_otsu, kernel, iterations=1)
    processed_images.append(("dilated", dilated))
    
    # Version 7: High-resolution version (2x scaling)
    scale_factor = 2
    high_res = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    _, high_res_binary = cv2.threshold(high_res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(("high_res", high_res_binary))
    
    # Run OCR on all processed versions and collect results
    all_ocr_results = []
    for name, img in processed_images:
        results = reader.readtext(img)
        all_ocr_results.extend([(r[0], r[1], r[2], name) for r in results])  # Include source image name
    
    # Deduplicate OCR results based on text content and confidence
    unique_text_segments = {}
    for bbox, text, conf, source in all_ocr_results:
        normalized_text = text.strip()
        if not normalized_text:
            continue
        
        # Keep the higher confidence result if we've seen this text before
        if normalized_text in unique_text_segments:
            if conf > unique_text_segments[normalized_text][1]:
                unique_text_segments[normalized_text] = (text, conf, bbox)
        else:
            unique_text_segments[normalized_text] = (text, conf, bbox)
    
    # Sort text segments by y-coordinate to preserve document flow
    sorted_segments = sorted(
        [(text, conf, bbox) for text, (_, conf, bbox) in unique_text_segments.items()],
        key=lambda x: (x[2][0][1] + x[2][2][1]) / 2  # Average y of top-left and bottom-right
    )
    
    # Extract full text in document order
    full_text = " ".join([segment[0] for segment in sorted_segments])
    
    # Run spaCy on the full text
    doc = nlp(full_text)
    
    # Initialize result dictionary
    entities = {
        "venue": "",
        "date": "",
        "time": "",
        "topic": "",
        "phone": [],
        "mode": "",
        "all_text": full_text  # Include all extracted text for debugging
    }
    
    # ENHANCED TOPIC DETECTION
    # Look for:
    # 1. All caps text
    # 2. Text with larger font size (implied by bounding box height)
    # 3. Text that appears at the top of the document
    
    # Get average text height
    if sorted_segments:
        text_heights = [(bbox[2][1] - bbox[0][1]) for _, _, bbox in sorted_segments]
        avg_height = sum(text_heights) / len(text_heights) if text_heights else 0
        
        topic_candidates = []
        
        # Top 3 segments are likely to contain the title/topic
        top_segments = sorted_segments[:3]
        for text, conf, bbox in top_segments:
            height = bbox[2][1] - bbox[0][1]
            # If text is in ALL CAPS or significantly larger than average
            if text.isupper() or height > (avg_height * 1.3):
                topic_candidates.append((text, conf))
        
        # Also look for any text that's much larger than average throughout the document
        for text, conf, bbox in sorted_segments:
            height = bbox[2][1] - bbox[0][1]
            if height > (avg_height * 1.5) and (text, conf) not in topic_candidates:
                topic_candidates.append((text, conf))
        
        # Sort by confidence and take the best
        if topic_candidates:
            topic_candidates.sort(key=lambda x: x[1], reverse=True)
            entities["topic"] = topic_candidates[0][0]
    
    # ENHANCED DATE DETECTION
    # Comprehensive date patterns
    date_patterns = [
        # Standard formats
        r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b',
        r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+\d{4}\b',
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b',
        r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b',
        # ISO format
        r'\b\d{4}-\d{2}-\d{2}\b',
        # Informal format
        r'\b(?:next|this|coming)\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
        # Month-Year format (partial date)
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b',
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'
    ]

    # Check for date patterns
    date_found = False
    for pattern in date_patterns:
        matches = re.finditer(pattern, full_text, re.IGNORECASE)
        for match in matches:
            try:
                date_text = match.group(0)
                date_obj = parser.parse(date_text, fuzzy=True)
                entities["date"] = date_obj.strftime("%Y-%m-%d")
                date_found = True
                break
            except:
                continue
        if date_found:
            break
    
    # Use NER as backup for date detection
    if not date_found:
        for ent in doc.ents:
            if ent.label_ == "DATE":
                try:
                    date_obj = parser.parse(ent.text, fuzzy=True)
                    # Only use if it looks like a complete date (not just month or day)
                    if date_obj.day and date_obj.month and date_obj.year:
                        entities["date"] = date_obj.strftime("%Y-%m-%d")
                        date_found = True
                        break
                except:
                    pass
    
    # ENHANCED TIME DETECTION
    time_patterns = [
        r'\b(?:\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm|AM|PM|a\.m\.|p\.m\.))\b',
        r'\b(?:\d{1,2}\s*(?:am|pm|AM|PM|a\.m\.|p\.m\.))\b',
        r'\b(?:\d{1,2}[-\s]?\d{1,2}\s*(?:hours|hrs))\b',
        r'\b(?:at|from|between)\s+\d{1,2}(?:[:.]?\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.|hours|hrs)?\b',
        r'\b\d{1,2}(?:[:.]?\d{2})?\s*(?:to|till|until|-)\s*\d{1,2}(?:[:.]?\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.|hours|hrs)?\b'
    ]
    
    # Check for time patterns
    time_found = False
    for pattern in time_patterns:
        matches = re.finditer(pattern, full_text, re.IGNORECASE)
        for match in matches:
            entities["time"] = match.group(0).strip()
            time_found = True
            break
        if time_found:
            break
    
    # Use NER as backup for time detection
    if not time_found:
        for ent in doc.ents:
            if ent.label_ == "TIME":
                entities["time"] = ent.text.strip()
                time_found = True
                break
    
    # ENHANCED VENUE DETECTION
    # Comprehensive venue keywords
    venue_keywords = [
        "conference hall", "auditorium", "room", "venue", "hall", "center", "centre",
        "building", "campus", "university", "college", "hotel", "theater", "theatre",
        "arena", "stadium", "ballroom", "gallery", "pavilion", "convention center",
        "community center", "expo", "fairground", "exhibition", "library", "museum",
        "institute", "plaza", "hall", "chamber", "location", "place", "address",
        "located at", "held at", "taking place at", "organized at", "hosted at"
    ]
    
    # First try to find venue using NER
    venue_candidates = []
    
    # Get entities that might be venues
    for ent in doc.ents:
        if ent.label_ in ["GPE", "ORG", "FAC", "LOC"]:
            venue_candidates.append((ent.text, 0.8))  # Base confidence score
    
    # Look for lines containing venue keywords
    for segment in sorted_segments:
        text = segment[0].lower()
        for keyword in venue_keywords:
            if keyword in text:
                # Get the full sentence or line containing the keyword
                venue_text = segment[0]
                confidence = segment[1]  # Use OCR confidence
                venue_candidates.append((venue_text, confidence))
                break
    
    # Find the venue with the highest confidence or most complete information
    if venue_candidates:
        # Sort by confidence and length (preferring longer, more complete venue descriptions)
        venue_candidates.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
        entities["venue"] = venue_candidates[0][0]
    
    # ENHANCED PHONE NUMBER DETECTION
    # More comprehensive phone patterns
    phone_patterns = [
        r'(?:\+\d{1,3}[-\.\s]?)?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4}',  # Standard format
        r'(?:\+\d{1,3}[-\.\s]?)?\d{5}[-\.\s]?\d{5}',  # 10 digits split in half
        r'(?:\+\d{1,3}[-\.\s]?)?\d{4}[-\.\s]?\d{3}[-\.\s]?\d{3}',  # Another split
        r'(?:phone|tel|contact|call)(?::|at|us)?[-\.\s]+(?:\+\d{1,3}[-\.\s]?)?\d+[-\.\s\d]+',  # With prefix
        r'(?:\+\d{1,3}[-\.\s]?)?(?:\(\d+\)|\d+)[-\.\s\d]{7,15}'  # Generic pattern
    ]
    
    # Find all phone matches across all patterns
    all_phones = []
    for pattern in phone_patterns:
        matches = re.finditer(pattern, full_text, re.IGNORECASE)
        for match in matches:
            phone = match.group(0).strip()
            # Clean up any noise
            phone = re.sub(r'[^\d\+\-\.\(\)\s]', '', phone)
            # Check it has enough digits to be a real phone number
            if sum(c.isdigit() for c in phone) >= 7:
                all_phones.append(phone)
    
    # Remove duplicates while preserving order
    seen = set()
    entities["phone"] = [x for x in all_phones if not (x in seen or seen.add(x))]
    
    # ENHANCED MODE DETECTION
    mode_keywords = {
        "online": [
            "online", "virtual", "zoom", "webinar", "stream", "digital", "remote", 
            "web conference", "video conference", "webex", "skype", "google meet",
            "microsoft teams", "livestream", "join online", "virtual event"
        ],
        "offline": [
            "in person", "in-person", "physical", "on site", "on-site", "face to face",
            "face-to-face", "attend in person", "attend at venue", "present at"
        ],
        "hybrid": [
            "hybrid", "both online and in person", "both virtual and physical",
            "online and offline", "in person and virtual", "join online or attend in person"
        ]
    }
    
    # Check for mode keywords in the text
    mode_scores = {"online": 0, "offline": 0, "hybrid": 0}
    
    for mode, keywords in mode_keywords.items():
        for keyword in keywords:
            if keyword in full_text.lower():
                mode_scores[mode] += 1
                
    # If "hybrid" has any matches, prioritize it
    if mode_scores["hybrid"] > 0:
        entities["mode"] = "hybrid"
    # Otherwise, take the mode with the most keyword matches
    else:
        max_score = max(mode_scores.values())
        if max_score > 0:
            for mode, score in mode_scores.items():
                if score == max_score:
                    entities["mode"] = mode
                    break
        # If no mode keywords found but we have a venue, assume offline
        elif entities["venue"]:
            entities["mode"] = "offline"
        # Default to online as a last resort
        else:
            entities["mode"] = "online"
            
    return entities

@app.post("/extract")
async def extract_info(file: UploadFile = File(...)):
    start_time = time.time()
    image_bytes = await file.read()
    result = process_image(image_bytes)
    end_time = time.time()
    result["processing_time"] = f"{end_time - start_time:.4f} seconds"
    return result
