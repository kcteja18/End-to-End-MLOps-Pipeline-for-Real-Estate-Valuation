import streamlit as st
import requests
import json

# --- App Title and Description ---
st.title("🏡 Real Estate Price Predictor")
st.write(
    "This app predicts the median house value for a district in California. "
    "Use the sliders and input boxes in the sidebar to set the features."
)

# --- Sidebar for User Inputs ---
st.sidebar.header("Input Housing Features")

def user_input_features():
    """Creates sidebar widgets and returns a dictionary of features."""
    med_inc = st.sidebar.slider("Median Income (in tens of thousands)", 2.0, 15.0, 8.3)
    house_age = st.sidebar.slider("House Age", 1.0, 52.0, 41.0)
    ave_rooms = st.sidebar.slider("Average Rooms", 2.0, 10.0, 6.9)
    ave_bedrms = st.sidebar.slider("Average Bedrooms", 1.0, 5.0, 1.0)
    population = st.sidebar.number_input("Population", min_value=3, max_value=37000, value=560)
    ave_occup = st.sidebar.slider("Average Occupancy", 1.0, 10.0, 3.0)
    latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 34.2)
    longitude = st.sidebar.slider("Longitude", -124.0, -114.0, -118.5)

    features = {
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude,
    }
    return features

input_features = user_input_features()

# Display the user inputs as a subheader
st.subheader("User Input Features")
st.json(input_features)

# --- Prediction Logic ---
if st.button("Predict"):
    # The URL of your running FastAPI application's predict endpoint
    api_url = "http://127.0.0.1:8000/predict/"

    # Send a POST request to the API
    response = requests.post(api_url, data=json.dumps(input_features))

    if response.status_code == 200:
        prediction = response.json()
        price = prediction["predicted_median_house_value"]
        st.success(f"**Predicted Median House Value:** ${price*100000:,.2f}")
    else:
        st.error("Error: Could not get a prediction from the API.")
        st.write("API Response:", response.text)