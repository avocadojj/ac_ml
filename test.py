import requests
import json

data = {'Age': 26.0, 'DependentChildren': 1.0, 'DependentsOther': 0.0, 'WeeklyWages': 600.18, 'HoursWorkedPerWeek': 40.0, 'DaysWorkedPerWeek': 5.0, 'InitialIncurredCalimsCost': 5300.0, 'DateTimeOfAccident_Year': 2002.0, 'DateTimeOfAccident_Month': 4.0, 'DateTimeOfAccident_Day': 2.0, 'DateReported_Year': 2002.0, 'DateReported_Month': 5.0, 'DateReported_Day': 7.0, 'CD_NECK': 0.0, 'CD_BACK': 0.0, 'CD_KNEE': 0.0, 'CD_FINGER': 0.0, 'CD_EYE': 0.0, 'CD_STRUCK': 0.0, 'CD_HAMMER': 0.0, 'CD_LADDER': 0.0, 'CD_STAIR': 0.0, 'CD_FELT': 0.0, 'CD_TRAUMA': 0.0, 'CD_FOREIGN_BODY': 0.0, 'CD_BACK_STRAIN': 0.0, 'CD_SOFT_TISSUE_': 0.0, 'CD_WORKPLACE_STRESS': 0.0, 'CD_LOWER_BACK_STRAIN': 0.0, 'CD_LEFT_RIGHT': 0.0, 'CD_LACERAT_': 0.0, 'Gender_0': 1.0, 'MaritalStatus_0': 1.0, 'MaritalStatus_1': 0.0, 'PartTimeFullTime_0': 0.0, 'Age_times_WeeklyWages': 15604.679999999998, 'and': 0.0, 'ankle': 0.0, 'arm': 0.0, 'back': 0.0, 'body': 0.0, 'bruised': 0.0, 'caught': 0.4434920495087984, 'cut': 0.0, 'elbow': 0.0, 'eye': 0.0, 'fell': 0.0, 'finger': 0.0, 'floor': 0.0, 'foot': 0.0, 'foreign': 0.0, 'from': 0.0, 'hand': 0.649144324760223, 'hit': 0.0, 'in': 0.0, 'index': 0.0, 'injury': 0.0, 'knee': 0.0, 'knife': 0.0, 'lacerated': 0.0, 'laceration': 0.0, 'left': 0.0, 'leg': 0.0, 'lifting': 0.0, 'lower': 0.0, 'metal': 0.0, 'middle': 0.0, 'neck': 0.0, 'of': 0.0, 'off': 0.0, 'on': 0.0, 'pain': 0.0, 'right': 0.4407918193100336, 'shoulder': 0.0, 'slipped': 0.0, 'soft': 0.0, 'sprained': 0.0, 'steel': 0.0, 'strain': 0.0, 'strained': 0.0, 'struck': 0.0, 'thumb': 0.0, 'tissue': 0.0, 'to': 0.4331616553706365, 'twisted': 0.0, 'wrist': 0.0, 'between': 0.0}


try:
    #response = requests.post('http://127.0.0.1:5000/predict', json=data)
    response = requests.post('https://uji-hehe-dot-gold-episode-394309.et.r.appspot.com/predict', json=data)
    response.raise_for_status()
    result = json.loads(response.text)
    print(f"Server Response: {result}")
    print(f'Predicted Ultimate Incurred Claim Cost: {result["prediction"]}')
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
except json.JSONDecodeError as e:
    print(f"Failed to decode JSON: {e}\nServer Response: {response.text}")
except KeyError as e:
    print(f"Key not found in response: {e}")



