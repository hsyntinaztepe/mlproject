import requests
from pymongo import MongoClient
import json


url = "https://1hfgp7cu95.execute-api.us-east-1.amazonaws.com/prod/api-ml-price-calculator"
headers = {"Content-Type": "application/json"}


payload = {
    "battery_power": 1454,
    "blue": 1,
    "clock_speed": 0.5,
    "dual_sim": 1,
    "fc": 1,
    "four_g": 0,
    "int_memory": 34,
    "m_dep": 0.7,
    "mobile_wt": 83,
    "n_cores": 4,
    "pc": 2,
    "px_height": 250,
    "px_width": 1033,
    "ram": 3419,
    "sc_h": 7,
    "sc_w": 5,
    "talk_time": 5,
    "three_g": 1,
    "touch_screen": 1,
    "wifi": 0
}


try:
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

  
    result = response.json()

    body = json.loads(result["body"])
    print("Prediction sonucu:", body)

except requests.exceptions.RequestException as e:
    print("Veri çekme hatası:", e)
    exit()


try:
    client = MongoClient("mongodb+srv://useradmin:admin@mlprojecluster.dl7if0o.mongodb.net/?retryWrites=true&w=majority&appName=mlprojecluster")
    db = client["awsmlproje"]
    collection = db["awsmlprojecollection"]
    print("MongoDB'ye bağlanıldı.")
except Exception as e:
    print("MongoDB bağlantı hatası:", e)
    exit()

try:
    payload_with_prediction = payload.copy()
    payload_with_prediction["prediction"] = body["prediction"][0]
    collection.insert_one(payload_with_prediction)
    print("Veri MongoDB'ye kaydedildi.")
except Exception as e:
    print("Veri kaydetme hatası:", e)
