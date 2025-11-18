import joblib
import gradio as gr
import numpy as np

pipeline = joblib.load("model.bin")

def predict_sleep_score(gender, age, occupation, sleep_duration,
                        physical_activity_level, stress_level, bmi_category,
                        heart_rate, daily_steps, sleep_disorder,
                        systolic_bp, diastolic_bp):

    sample = {
        "gender": gender,
        "age": int(age),
        "occupation": occupation,
        "sleep_duration": float(sleep_duration),
        "physical_activity_level": int(physical_activity_level),
        "stress_level": int(stress_level),
        "bmi_category": bmi_category,
        "heart_rate": int(heart_rate),
        "daily_steps": int(daily_steps),
        "sleep_disorder": sleep_disorder,
        "systolic_bp": int(systolic_bp),
        "diastolic_bp": int(diastolic_bp)
    }

    pred = pipeline.predict(sample)
    return int(pred)
    

with gr.Blocks() as app:
    gr.Markdown("# 💤 Sleep Quality Score Predictor")

    with gr.Row():
        gender = gr.Dropdown(["male", "female"], label="Gender")
        age = gr.Number(label="Age", value=30)
        occupation = gr.Dropdown(
            [
                "Accountant", "Doctor", "Engineer", "Lawyer", "Manager", "Nurse",
                "Sales Representative", "Salesperson", "Scientist",
                "Software Engineer", "Teacher"
            ],
            label="Occupation"
        )

    with gr.Row():
        sleep_duration = gr.Number(label="Sleep Duration (hours)")
        physical_activity_level = gr.Number(label="Physical Activity Level")
        stress_level = gr.Number(label="Stress Level (0–10)")

    with gr.Row():
        bmi_category = gr.Dropdown(
            ["Normal", "Normal Weight", "Overweight", "Obese"],
            label="BMI Category"
        )
        heart_rate = gr.Number(label="Heart Rate")
        daily_steps = gr.Number(label="Daily Steps")

    with gr.Row():
        sleep_disorder = gr.Dropdown(
            ["Insomnia", "None", "Sleep Apnea"], 
            label="Sleep Disorder"
        )
        systolic_bp = gr.Number(label="Systolic BP")
        diastolic_bp = gr.Number(label="Diastolic BP")

    output = gr.Number(label="Predicted Sleep Score")

    submit = gr.Button("Predict Score")
    submit.click(
        predict_sleep_score,
        inputs=[
            gender, age, occupation, sleep_duration,
            physical_activity_level, stress_level, bmi_category,
            heart_rate, daily_steps, sleep_disorder,
            systolic_bp, diastolic_bp
        ],
        outputs=output
    )

app.launch()
