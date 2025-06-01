
from pymongo.mongo_client import MongoClient
import os
from dotenv import load_dotenv
from pathlib import Path

# Load the .env file from parent directory
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

username = os.getenv("MONGO_USERNAME")
password = os.getenv("MONGO_PASSWORD")

uri = f"mongodb+srv://{username}:{password}@cluster0.xxvysv5.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri)

mongodb = client["drivers"]

def log_drivers_stats_to_mongo(
    date: str,
    driver_name: str,
    warehouse: str,
    route: str,
    successful_deliveries: int,
    successful_collections: int,
    image_link: str
 ) -> bool:
    """
    Logs driver data into MongoDB. If the collection (named after the warehouse) does not exist, 
    MongoDB automatically creates it.
    """
    try:
        data = {
            "date": date,
            "driver_name": driver_name,
            "route": route,
            "successful_deliveries": successful_deliveries,
            "successful_collections": successful_collections,
            "total_jobs": successful_deliveries + successful_collections,
            "image_link": image_link
        }

        collection = mongodb[warehouse]
        collection.insert_one(data)

        print(f"Log inserted successfully into {warehouse} collection")
        return True
    except Exception as e:
        print(f"Mongo DB logged failed : {e}")
        return False
    
