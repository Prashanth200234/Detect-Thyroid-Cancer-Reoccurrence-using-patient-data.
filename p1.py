import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "Age": 65,
    "Gender": "Male",
    "Smoking": "Yes",
    "Hx_Smoking": "Yes",
    "Hx_Radiotherapy": "Yes",
    "Thyroid_Function": "Abnormal",
    "Physical_Examination": "Abnormal",
    "Adenopathy": "Yes",
    "Pathology": "Anaplastic",
    "Focality": "Unifocal",
    "Risk": "High",
    "T": 4,
    "N": 4,
    "M": 6,
    "Stage": 4,
    "Response": "Non-Responsive"
}

response = requests.post(url, json=data)
print(response.json())
