import requests

url = 'http://localhost:9696/predict'

user = {
    'gender': 'female',
    'age': 35,
    'occupation': "Doctor",
    'sleep_duration': 7.5,
    'quality_of_sleep': 85,
    'physical_activity_level': 3,
    'stress_level': 2,
    'bmi_category': "Overweight",
    'heart_rate': 72,
    'daily_steps': 8000,
    'sleep_disorder': "None",
    'systolic_bp': 120,
    'diastolic_bp': 80
}


response = requests.post(url, json=user)

predictions = response.json()

if predictions['sleep_score'] > 6:
    print(f'Your sleep score is {predictions["sleep_score"]}, quite good')
else:
    print(f'Your sleep score is {predictions["sleep_score"]}, you need to improve')