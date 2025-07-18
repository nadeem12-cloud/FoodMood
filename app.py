import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("model/recommender.pkl")
le_mood = joblib.load("model/le_mood.pkl")
le_time = joblib.load("model/le_time.pkl")
le_weather = joblib.load("model/le_weather.pkl")
le_food = joblib.load("model/le_food.pkl")

# App Title
st.set_page_config(page_title="FoodMood 🍽️", layout="centered")
st.title("🍽️ FoodMood: AI-Based Mood Food Recommender")
st.markdown("Feeling hungry? Tell me your mood, and I’ll serve you the vibe!")

# Sidebar Inputs
mood = st.selectbox("How are you feeling today?", le_mood.classes_)
time_of_day = st.selectbox("What's the time of day?", le_time.classes_)
weather = st.selectbox("What's the weather like?", le_weather.classes_)

# Prediction
if st.button("Get My Food! 🚀"):
    try:
        mood_enc = le_mood.transform([mood])[0]
        time_enc = le_time.transform([time_of_day])[0]
        weather_enc = le_weather.transform([weather])[0]

        pred = model.predict([[mood_enc, time_enc, weather_enc]])
        recommended_food = le_food.inverse_transform(pred)[0]

        st.success(f"🍱 You should try: **{recommended_food}**")
        st.balloons()

    except Exception as e:
        st.error("Oops! Something went wrong. Please check your inputs.")

# Footer
st.markdown("---")
st.markdown("Made with 💛 by Nadeem | #FoodMoodAI")
