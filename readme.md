### OCR & NLP Processing API

### This API extracts structured information from images containing text using EasyOCR and spaCy. The extracted details include venue, date, time, topic, phone numbers, and event mode (online/offline). It is built using FastAPI and can process uploaded images to return structured JSON data.###

Features

Extracts text from images using OCR.

Identifies key details such as venue, date, time, topic, and phone numbers.

Determines event mode (online, offline, or hybrid) based on extracted information.

Returns structured data in JSON format.

Requirements

Python 3.11.9
Explicitly use python 3.10 for better alignment with tenacity and mega


### Create a Virtual Environment\
```
python3.10 -m venv .venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  
```

### Install Dependencies
```
pip install -r requirements.txt
```

### Download spaCy Model
```
python -m spacy download en_core_web_trf
```

### Running the API
```
uvicorn mes:app --host 0.0.0.0 --port 8000 --reload

# to run the client backend
source .venv/bin/activate
uvicorn new:app --host 0.0.0.0 --port 8000 --reload 
```

# to use ngrok for the backend URL
before starting the ngrok make sure that server is running
brew install ngrok
ngrok http 8000

# you can start server also on 
ngrok http http://localhost:8080

## Testing the API using cURL
```
curl -X 'POST' \
  'http://127.0.0.1:8000/extract' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/image.jpg'
  ```


### Author

### Developed by Swapnil Sonker.