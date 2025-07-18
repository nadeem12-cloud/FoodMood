import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

# Load the dataset
df = pd.read_csv("data/mood_food_data.csv")

# Initialize encoders
le_mood = LabelEncoder()
le_time = LabelEncoder()
le_weather = LabelEncoder()
le_food = LabelEncoder()

# Encode the categorical columns
df['mood_enc'] = le_mood.fit_transform(df['mood'])
df['time_enc'] = le_time.fit_transform(df['time_of_day'])
df['weather_enc'] = le_weather.fit_transform(df['weather'])
df['food_enc'] = le_food.fit_transform(df['recommended_food'])

# Features and target
X = df[['mood_enc', 'time_enc', 'weather_enc']]
y = df['food_enc']

# Train a Decision Tree model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model and encoders
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/recommender.pkl")
joblib.dump(le_mood, "model/le_mood.pkl")
joblib.dump(le_time, "model/le_time.pkl")
joblib.dump(le_weather, "model/le_weather.pkl")
joblib.dump(le_food, "model/le_food.pkl")

print("✅ Model and encoders trained and saved successfully!")
