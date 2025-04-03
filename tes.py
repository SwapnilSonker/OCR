from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import numpy as np
import cv2
import re
from typing import List
import easyocr
from datetime import datetime
import spacy
import dateutil.parser as dparser
from PIL import Image
import io
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Event Image Extraction API",
    description="API to extract event information from promotional images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Initialize EasyOCR reader at startup
@app.on_event("startup")
async def startup_event():
    global reader, nlp
    try:
        # Initialize EasyOCR with English language
        reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=False if no GPU available
        logger.info("Initialized EasyOCR reader with GPU support")
    except Exception as e:
        reader = easyocr.Reader(['en'], gpu=False)
        logger.info(f"Initialized EasyOCR reader without GPU: {str(e)}")
    
    try:
        nlp = spacy.load('en_core_web_trf')
        logger.info("Loaded transformer-based spaCy model")
    except OSError:
        nlp = spacy.load('en_core_web_sm')
        logger.info("Loaded small spaCy model (transformer model not available)")

def process_event_image(image_bytes, verbose=False):
    """
    Process an event image to extract structured information using EasyOCR.
    
    Args:
        image_bytes: Image data as bytes
        verbose: Set to True to print debugging information
        
    Returns:
        dict: Extracted event information
    """
    # Convert image bytes to OpenCV format
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image")
        
        # Store original image dimensions
        height, width = image.shape[:2]
        
        # Initialize result structure
        event_info = {
            "topic": "",
            "date": "",
            "time": "",
            "venue": "",
            "mode": "",
            "contact": "",
            "guest": "",
            "all_text": ""
        }
        
        # Create multiple processed versions of the image for better OCR results
        processed_images = []
        
        # Basic image - original RGB
        processed_images.append(("original", image))
        
        # Basic preprocessing - grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_images.append(("gray", gray))
        
        # High contrast version
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        processed_images.append(("enhanced", enhanced))
        
        # Extract text from each processed image
        full_text = ""
        text_blocks = []
        
        # Use EasyOCR to extract text from the images
        for name, img in processed_images:
            try:
                # EasyOCR provides bounding boxes, text and confidence scores
                results = reader.readtext(img)
                
                for detection in results:
                    bbox, text, confidence = detection
                    
                    # Only include text with reasonable confidence
                    if confidence > 0.5 and text.strip():
                        full_text += text + " "
                        
                        # Calculate coordinates and dimensions
                        x_min = min(point[0] for point in bbox)
                        y_min = min(point[1] for point in bbox)
                        x_max = max(point[0] for point in bbox)
                        y_max = max(point[1] for point in bbox)
                        w = x_max - x_min
                        h = y_max - y_min
                        
                        # Store text block with position and size information
                        text_blocks.append({
                            'text': text,
                            'x': int(x_min),
                            'y': int(y_min),
                            'width': int(w),
                            'height': int(h),
                            'area': int(w * h),
                            'conf': confidence,
                            'source': name
                        })
            except Exception as e:
                logger.warning(f"Error processing {name} image: {str(e)}")
        
        # Clean and normalize full text
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        event_info['all_text'] = full_text
        
        # Use NLP to process the text
        doc = nlp(full_text[:nlp.max_length])  # Limit to max length supported by the model
        
        # Function to clean up extracted text
        def clean_text(text):
            return re.sub(r'\s+', ' ', text).strip()
        
        # --------- TOPIC EXTRACTION ---------
        # 1. Look for large text blocks (typically titles/headlines)
        # 2. Look for text in the upper portion of the image
        # 3. Look for all caps text
        
        # Sort text blocks by size (area) and filter to top third of image
        large_blocks = [b for b in text_blocks if b['y'] < (height / 3) and b['area'] > 1000]
        large_blocks.sort(key=lambda x: x['area'], reverse=True)
        
        # Also look for all caps text that might be titles
        caps_blocks = [b for b in text_blocks if b['text'].isupper() and len(b['text']) > 3]
        
        topic_candidates = []
        
        # Add large blocks from top third of image
        for block in large_blocks[:3]:  # Consider top 3 large blocks
            topic_candidates.append((block['text'], block['area'], block['conf']))
        
        # Add all caps blocks
        for block in caps_blocks:
            if any(block['text'] in cand[0] for cand in topic_candidates):
                continue  # Skip if already included
            topic_candidates.append((block['text'], block['area'], block['conf']))
        
        # Prioritize by size, then confidence
        if topic_candidates:
            topic_candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
            # Clean up potential topic
            topic = topic_candidates[0][0]
            
            # Look for pattern like "SOMETHING presents" and exclude it
            if "present" in topic.lower():
                parts = re.split(r'\s+presents\s+', topic, flags=re.IGNORECASE)
                if len(parts) > 1:
                    topic = parts[1]  # Take what comes after "presents"
            
            event_info['topic'] = clean_text(topic)
        
        # If we still don't have a topic, look for potential phrases in the text
        if not event_info['topic']:
            potential_phrases = [
                "awareness programme", "conference", "seminar", "workshop", 
                "event", "celebration", "day", "festival", "symposium"
            ]
            
            for phrase in potential_phrases:
                match = re.search(rf'(?i)([A-Z][A-Za-z\s]+\s+{phrase})', full_text)
                if match:
                    event_info['topic'] = clean_text(match.group(1))
                    break
        
        # --------- DATE EXTRACTION ---------
        date_patterns = [
            r'\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b',
            r'\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)\b',
            r'\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b',
            r'\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))\b',
            r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b'
        ]
        
        # Try to find dates in text
        for pattern in date_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                try:
                    date_text = match.group(0)
                    # Try to parse the date
                    date_obj = dparser.parse(date_text, fuzzy=True)
                    event_info['date'] = date_obj.strftime("%Y-%m-%d")
                    break
                except:
                    continue
            if event_info['date']:
                break
        
        # If no date found via regex, try NER
        if not event_info['date']:
            for ent in doc.ents:
                if ent.label_ == "DATE":
                    try:
                        # Exclude vague dates like "today", "tomorrow", etc.
                        vague_terms = ["today", "tomorrow", "yesterday", "next week", "last week"]
                        if not any(term in ent.text.lower() for term in vague_terms):
                            date_obj = dparser.parse(ent.text, fuzzy=True)
                            event_info['date'] = date_obj.strftime("%Y-%m-%d")
                            break
                    except:
                        pass
        
        # --------- TIME EXTRACTION ---------
        time_patterns = [
            r'\b(?:\d{1,2}:\d{2}\s*(?:am|pm|AM|PM|a\.m\.|p\.m\.))\b',
            r'\b(?:\d{1,2}\s*(?:am|pm|AM|PM|a\.m\.|p\.m\.))\b',
            r'\b(?:at|from)\s+\d{1,2}(?:[:.]?\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.|hours|hrs)?\b',
            r'\b\d{1,2}(?:[:.]?\d{2})?\s*(?:to|till|until|-)\s*\d{1,2}(?:[:.]?\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.|hours|hrs)?\b'
        ]
        
        for pattern in time_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                event_info['time'] = clean_text(match.group(0))
                break
            if event_info['time']:
                break
        
        # --------- VENUE/ADDRESS EXTRACTION ---------
        # Keywords that might indicate a venue/location
        venue_indicators = [
            "venue", "location", "place", "address", "at", "held at", "taking place at",
            "conference hall", "auditorium", "hall", "center", "centre", "building", 
            "campus", "university", "hotel", "room", "street", "road", "avenue"
        ]

        # Address-like regex patterns
        address_patterns = [
            r'\d{1,5}\s\w+\s(?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Lane|Ln|Drive|Dr)',  # "123 Main Street"
            r'\b(?:PO Box|P\.O\. Box|Suite|Apt|Apartment)\s\d+',  # "PO Box 123", "Suite 456"
            # r'\b[A-Za-z\s]+,\s?[A-Za-z\s]+,\s?\d{5}(?:-\d{4})?',  # "New York, NY 10001"
            # r'\b[A-Za-z\s]+,\s?[A-Za-z\s]+',  # "Los Angeles, CA"
            # r'\d{1,5}\s[A-Za-z\s]+,\s?[A-Za-z\s]+,\s?\d{5}'  # "123 Elm St, Springfield, 62704"
        ]

        venue_candidates = []
        address_found = False  # Track if a valid address exists

        doc = nlp(full_text)

        # Try NER for locations first
        for ent in doc.ents:
            if ent.label_ in ["GPE", "ORG", "FAC", "LOC"]:
                if not any(ent.text.lower() in date.lower() for date in [event_info['date'], event_info['time']]):
                    venue_candidates.append(ent.text)

        # Look for phrases containing venue indicators
        for indicator in venue_indicators:
            pattern = re.compile(rf'(?i)(?:{indicator}\s*[:;]?\s*)([^\.!?\n]+)', re.IGNORECASE)
            matches = pattern.finditer(full_text)
            for match in matches:
                venue_text = match.group(1).strip()
                if venue_text and venue_text.lower() != indicator.lower():
                    venue_candidates.append(venue_text)

        # Look for explicit address patterns in the full text
        for pattern in address_patterns:
            matches = re.findall(pattern, full_text)
            if matches:
                venue_candidates.extend(matches)
                address_found = True  # Mark that an address was found

        # Clean up and select the most likely venue/address
        if venue_candidates:
            venue_candidates.sort(key=len, reverse=True)
            event_info['venue'] = clean_text(venue_candidates[0])

        # --------- MODE DETECTION ---------
        # Determine if event is online, offline, or hybrid
        online_indicators = ["online", "virtual", "zoom", "webinar", "teams", "digital", "web", "stream"]
        offline_indicators = ["in person", "in-person", "venue", "physical", "on-site", "location"]
        hybrid_indicators = ["hybrid", "both online and offline", "online and in-person"]

        mode_scores = {"online": 0, "offline": 0, "hybrid": 0}

        # Count indicators in the text
        for indicator in online_indicators:
            if indicator in full_text.lower():
                mode_scores["online"] += 1

        for indicator in offline_indicators:
            if indicator in full_text.lower():
                mode_scores["offline"] += 1

        for indicator in hybrid_indicators:
            if indicator in full_text.lower():
                mode_scores["hybrid"] += 1

        # Final mode determination
        if mode_scores["hybrid"] > 0:
            event_info["mode"] = "hybrid"
        elif address_found:  # If an address is found, assume offline
            event_info["mode"] = "offline"
        elif mode_scores["online"] > mode_scores["offline"]:
            event_info["mode"] = "online"
        elif mode_scores["offline"] > mode_scores["online"]:
            event_info["mode"] = "offline"
        elif event_info["venue"]:  # If venue exists, assume offline
            event_info["mode"] = "offline"
        else:
            event_info["mode"] = "online"  # Default to online if nothing is found

        
        # --------- CONTACT INFORMATION ---------
        # Look for phone numbers
        phone_patterns = [
            # Matches numbers like +123-345-345, +91-12345-67890, +44-203-123-4567, +9845 0839 938
            r'\+\d{1,4}[-.\s]?\d{2,5}[-.\s]?\d{2,5}[-.\s]?\d{2,5}',  

            # Matches standard international formats like +1 (123) 456-7890 or +91 98765 43210
            r'\+\d{1,4}[-.\s]?\(?\d{2,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5}',  

            # Matches numbers with or without country codes like (123) 456-7890 or 123-456-7890
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  

            # Matches longer variations like +123 4567 89012
            r'\+\d{1,4}[-.\s]?\d{4,6}[-.\s]?\d{4,6}', 
        ]
        
        for pattern in phone_patterns:
            matches = re.finditer(pattern, full_text)
            for match in matches:
                event_info['contact'] = match.group(0).strip()
                break
            if event_info['contact']:
                break
        
        # --------- GUEST SPEAKER DETECTION ---------
        guest_indicators = ["guest", "speaker", "presented by", "presenter", "dr", "dr.", "prof", "prof."]
        
        # Look for guest speaker patterns
        for indicator in guest_indicators:
            pattern = re.compile(rf'(?i)(?:{indicator}\s*[:;]?\s*)([^\.!?\n,]+)', re.IGNORECASE)
            matches = pattern.finditer(full_text)
            for match in matches:
                guest_text = match.group(1).strip()
                if guest_text and len(guest_text) > 3:  # Simple validation
                    event_info['guest'] = clean_text(guest_text)
                    break
            if event_info['guest']:
                break
        
        # Also look for "Dr. Name" or "Prof. Name" patterns
        if not event_info['guest']:
            doctor_pattern = re.compile(r'(?i)(?:dr\.?|prof\.?|professor)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)')
            matches = doctor_pattern.finditer(full_text)
            for match in matches:
                event_info['guest'] = clean_text(match.group(0))
                break
        
        # Final cleanup of the event info
        for key in event_info:
            if isinstance(event_info[key], str):
                event_info[key] = clean_text(event_info[key])
        
        return event_info
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise ValueError(f"Image processing failed: {str(e)}")

@app.post("/extract")
async def extract_info(files: List[UploadFile] = File(...)):
    """
    Extract information from an event poster image
    
    Args:
        file: The uploaded image file
        
    Returns:
        dict: Extracted event information with processing time
    """
    try:
        results = []
        for file in files:
        # Record start time for performance measurement
            start_time = time.time()
            
            # Check file type
            content_type = file.content_type
            if not content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            # Read the image bytes
            image_bytes = await file.read()
            
            # Process the image
            result = process_event_image(image_bytes)
            
            # Calculate processing time
            end_time = time.time()
            result["processing_time"] = f"{end_time - start_time:.4f} seconds"
            
            results.append(result)
            
            return {"extracted_data":results}
    
    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during processing")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok", "message": "API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)