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
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

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

# Global variables for models
reader = None
nlp = None
executor = None

# Initialize EasyOCR reader at startup
@app.on_event("startup")
async def startup_event():
    global reader, nlp, executor
    try:
        # Initialize EasyOCR with English language - with lower beam width for faster processing
        reader = easyocr.Reader(['en'], gpu=False, beamWidth=3, batch_size=4)
        logger.info("Initialized EasyOCR reader with optimized parameters")
    except Exception as e:
        reader = easyocr.Reader(['en'], gpu=False)
        logger.info(f"Initialized EasyOCR reader without GPU: {str(e)}")
    
    try:
        # Use smaller model by default for speed
        nlp = spacy.load('en_core_web_sm')
        logger.info("Loaded small spaCy model for better performance")
    except OSError:
        try:
            nlp = spacy.load('en_core_web_sm')
            logger.info("Loaded transformer-based spaCy model")
        except:
            logger.error("Failed to load any spaCy model")
            raise
    
    # Initialize thread pool for parallel processing
    executor = ThreadPoolExecutor(max_workers=4)
    logger.info("Initialized thread pool executor")

@app.on_event("shutdown")
async def shutdown_event():
    global executor
    if executor:
        executor.shutdown()
        logger.info("Shut down thread pool executor")

@lru_cache(maxsize=32)
def parse_date(date_text):
    """Cache date parsing to avoid repetitive parsing of the same date text"""
    try:
        date_obj = dparser.parse(date_text, fuzzy=True)
        return date_obj.strftime("%Y-%m-%d")
    except:
        return None

def process_image_version(img_data):
    """Process a single image version with EasyOCR"""
    name, img = img_data
    try:
        # More efficient parameter setting for EasyOCR
        # Set lower threshold, paragraph=False for faster processing
        results = reader.readtext(img, paragraph=False, min_size=10, text_threshold=0.5, link_threshold=0.4, low_text=0.4)
        
        text_blocks = []
        for detection in results:
            bbox, text, confidence = detection
            
            # Only include text with reasonable confidence
            if confidence > 0.4 and text.strip():  # Lower threshold for more speed
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
        return text_blocks
    except Exception as e:
        logger.warning(f"Error processing {name} image: {str(e)}")
        return []

def process_event_image(image_bytes, verbose=False):
    """
    Process an event image to extract structured information using EasyOCR.
    
    Args:
        image_bytes: Image data as bytes
        verbose: Set to True to print debugging information
        
    Returns:
        dict: Extracted event information
    """
    global executor
    
    # Convert image bytes to OpenCV format
    try:
        # Decode image more efficiently
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image")
        
        # Resize large images to reduce processing time
        max_dimension = 1200
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            height, width = image.shape[:2]
            logger.info(f"Resized image to {width}x{height} for faster processing")
        
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
        # Limit to 2 versions for optimization
        processed_images = []
        
        # Original RGB - skip if we want to save time
        processed_images.append(("original", image))
        
        # Basic preprocessing - grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use CLAHE only on grayscale image for efficiency
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        processed_images.append(("enhanced", enhanced))
        
        # Extract text from each processed image in parallel
        full_text = ""
        text_blocks = []
        
        # Use parallel processing for OCR on different image versions
        results = list(executor.map(process_image_version, processed_images))
        
        # Combine all text blocks
        for blocks in results:
            text_blocks.extend(blocks)
        
        # Combine all text
        for block in text_blocks:
            full_text += block['text'] + " "
        
        # Clean and normalize full text
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        event_info['all_text'] = full_text
        
        # Function to clean up extracted text
        def clean_text(text):
            return re.sub(r'\s+', ' ', text).strip()
        
        # --------- TOPIC EXTRACTION ---------
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
        # Precompile regex patterns for better performance
        date_patterns = [
            re.compile(r'\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b', re.IGNORECASE),
            re.compile(r'\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)\b', re.IGNORECASE),
            re.compile(r'\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b', re.IGNORECASE),
            re.compile(r'\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))\b', re.IGNORECASE),
            re.compile(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b'),
            re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b', re.IGNORECASE),
            re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b', re.IGNORECASE)
        ]
        
        # Try to find dates in text
        for pattern in date_patterns:
            matches = pattern.finditer(full_text)
            for match in matches:
                date_text = match.group(0)
                parsed_date = parse_date(date_text)
                if parsed_date:
                    event_info['date'] = parsed_date
                    break
            if event_info['date']:
                break
        
        # If no date found via regex, try lightweight NER processing
        if not event_info['date'] and len(full_text) < 5000:  # Only process reasonably sized text
            # Use a smaller chunk of text for NER to improve speed
            # Process only first 1000 characters for date extraction
            doc = nlp(full_text[:min(1000, len(full_text))])
            for ent in doc.ents:
                if ent.label_ == "DATE":
                    # Exclude vague dates like "today", "tomorrow", etc.
                    vague_terms = ["today", "tomorrow", "yesterday", "next week", "last week"]
                    if not any(term in ent.text.lower() for term in vague_terms):
                        parsed_date = parse_date(ent.text)
                        if parsed_date:
                            event_info['date'] = parsed_date
                            break
        
        # --------- TIME EXTRACTION ---------
        # Precompile time patterns
        time_patterns = [
            # Matches a single time with optional AM/PM (e.g., 6am, 6:30pm, 6 AM, 6:30 PM)
            re.compile(r'\b(?:\d{1,2}[:.]?\d{2})\s*(?:am|pm|AM|PM|a\.m\.|p\.m\.|a\.m|p\.m)\b', re.IGNORECASE),
            
            # Matches single times with AM/PM (e.g., 6am, 6 AM)
            re.compile(r'\b(?:\d{1,2})\s*(?:am|pm|AM|PM|a\.m\.|p\.m\.|a\.m|p\.m)\b', re.IGNORECASE),
            
            # Matches times with 'at', 'from' (e.g., at 6am, from 6:30pm, from 8:00 AM)
            re.compile(r'\b(?:at|from)\s+\d{1,2}(?:[:.]?\d{2})?\s*(?:am|pm|AM|PM|a\.m\.|p\.m\.|a\.m|p\.m)?\b', re.IGNORECASE),
            
            # Matches time ranges like '6am-8pm', '6:30am to 8:30pm', '6am till 8pm', '6am to 8pm'
            re.compile(r'\b(?:\d{1,2}[:.]?\d{2})\s*(?:am|pm|AM|PM|a\.m\.|p\.m\.|a\.m|p\.m)\s*(?:-|to|till|until)\s*\d{1,2}[:.]?\d{2}\s*(?:am|pm|AM|PM|a\.m\.|p\.m\.|a\.m|p\.m)\b', re.IGNORECASE),
            
            # Matches time ranges like '6am-8pm', '4am to 8pm', '4am till 8pm' (without spaces)
            re.compile(r'\b(?:\d{1,2})\s*(?:am|pm|AM|PM|a\.m\.|p\.m)\s*(?:-|to|till|until)?\s*\d{1,2}\s*(?:am|pm|AM|PM|a\.m\.|p\.m)\b', re.IGNORECASE),
            
            # Matches time ranges without space (e.g., 6amto8pm, 4amto8pm)
            re.compile(r'\b(?:\d{1,2})\s*(?:am|pm|AM|PM|a\.m\.|p\.m)\s*(?:to|till|until)?\s*(?:\d{1,2})\s*(?:am|pm|AM|PM|a\.m\.|p\.m)\b', re.IGNORECASE),

            # Matches time ranges with no spaces, minutes, and AM/PM (e.g., 6:30am-8:30pm, 6:30am to 8:30pm)
            re.compile(r'\b(?:\d{1,2}[:.]?\d{2})\s*(?:am|pm|AM|PM|a\.m\.|p\.m)\s*(?:-|to|till|until)?\s*(?:\d{1,2}[:.]?\d{2})\s*(?:am|pm|AM|PM|a\.m\.|p\.m)\b', re.IGNORECASE),
            
            # Match time ranges like '6am to 8pm', '6am-8pm' including cases with multiple spaces
            re.compile(r'\b(?:\d{1,2})\s*(?:am|pm|AM|PM|a\.m\.|p\.m)\s*(?:\s*[-to\s*]*|[-\s*]*)(?:\d{1,2})\s*(?:am|pm|AM|PM|a\.m\.|p\.m)\b', re.IGNORECASE)
        ]
        
        for pattern in time_patterns:
            matches = pattern.finditer(full_text)
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

        # Address-like regex patterns - precompiled
        address_patterns = [
            re.compile(r'\d{1,5}\s\w+\s(?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Lane|Ln|Drive|Dr)'),
            re.compile(r'\b(?:PO Box|P\.O\. Box|Suite|Apt|Apartment)\s\d+')
        ]

        venue_candidates = []
        address_found = False  # Track if a valid address exists

        # Process only a portion of text for NER to save time
        if len(full_text) < 3000:  # Only process reasonably sized text
            doc = nlp(full_text[:min(2000, len(full_text))])
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
                    break  # Get just one match per indicator for efficiency

        # Look for explicit address patterns in the full text
        for pattern in address_patterns:
            matches = pattern.findall(full_text)
            if matches:
                venue_candidates.extend(matches[:1])  # Just get the first match for efficiency
                address_found = True  # Mark that an address was found
                break

        # Clean up and select the most likely venue/address
        if venue_candidates:
            venue_candidates.sort(key=len, reverse=True)
            event_info['venue'] = clean_text(venue_candidates[0])

        # --------- MODE DETECTION ---------
        # Faster mode detection with simple keyword counting
        if "hybrid" in full_text.lower():
            event_info["mode"] = "hybrid"
        elif address_found:  # If an address is found, assume offline
            event_info["mode"] = "offline"
        elif any(term in full_text.lower() for term in ["online", "virtual", "zoom", "webinar", "teams"]):
            event_info["mode"] = "online"
        elif any(term in full_text.lower() for term in ["in person", "in-person", "venue", "physical"]):
            event_info["mode"] = "offline"
        elif event_info["venue"]:  # If venue exists, assume offline
            event_info["mode"] = "offline"
        else:
            event_info["mode"] = "online"  # Default to online if nothing is found
        
        # --------- CONTACT INFORMATION ---------
        # Precompile phone patterns
        phone_patterns = [
            # International phone number with optional country code, parentheses, and separators (hyphen, dot, space)
            re.compile(r'\+\d{1,4}[-.\s]?\(?\d{1,5}\)?[-.\s]?\d{1,5}[-.\s]?\d{1,5}'),  # +<code> (xxx) xxx xxx or +<code> xxx xxx xxx
            
            # Standard phone number format with optional parentheses around the area code
            re.compile(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),  # (xxx) xxx-xxxx or xxx-xxx-xxxx
            
            # International phone number with country code and an optional area code in parentheses
            re.compile(r'\+\d{1,4}[-.\s]?\(?\d{2,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5}'),  # +<code> (xxx) xxx xxx or +<code> xxx xxx xxx
            
            # Simple phone number with no parentheses or country code (local format)
            re.compile(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{3}'),  # 222 333 333 or 222-333-333 or 222.333.333
            
            # Another format for phone numbers with parentheses around the area code and optional separators
            re.compile(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{3}'),  # (333) 333 333
            
            # Optional international phone numbers with or without separators and area code (e.g., +1 555 123 4567)
            re.compile(r'\+\d{1,4}[-.\s]?\(?\d{1,5}\)?[-.\s]?\d{1,5}[-.\s]?\d{1,5}'),  # +1 555 123 4567 or +1(555)1234567
            
            # Match numbers with 4-digit area codes with optional separators
            re.compile(r'\(?\d{4}\)?[-.\s]?\d{3}[-.\s]?\d{3}'),  # (5555) 123-123 or 5555-123-123
            
            # Match short codes for SMS or other short phone numbers
            re.compile(r'\d{5,6}'),  # 12345 or 123456 (for short codes)
            
            # Match vanity phone numbers (e.g., 1-800-FLOWERS, where the number can be represented by letters)
            re.compile(r'1-\d{3}-[A-Z]{3,4}'),  # Match numbers like 1-800-FLOWERS (letters replaced with digits in real cases)
        ]
        
        for pattern in phone_patterns:
            matches = pattern.finditer(full_text)
            for match in matches:
                event_info['contact'] = match.group(0).strip()
                break
            if event_info['contact']:
                break
        
        # --------- GUEST SPEAKER DETECTION ---------
        # Just check for basic patterns instead of multiple searches
        guest_pattern = re.compile(r'(?i)(?:guest|speaker|presented by|presenter|dr\.?|prof\.?|professor)\s*[:;]?\s*([^\.!?\n,]+)')
        matches = guest_pattern.finditer(full_text)
        for match in matches:
            guest_text = match.group(1).strip()
            if guest_text and len(guest_text) > 3:  # Simple validation
                event_info['guest'] = clean_text(guest_text)
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
        
        return {"extracted_data": results}
    
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