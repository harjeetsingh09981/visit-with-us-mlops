
import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "tourism_project/deployment/best_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("Wellness Tourism Package — Purchase Prediction")
st.write(
    "Enter a customer's details below to predict whether they are "
    "likely to purchase the Wellness Tourism Package."
)

with st.form("prediction_form"):
    st.subheader("Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=61, value=35)
        type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        gender = st.selectbox("Gender", ["Male", "Female", "Fe Male"])
        number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=5, value=2)
        number_of_followups = st.number_input("Number of Followups", min_value=1.0, max_value=6.0, value=3.0)
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])

    with col2:
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
        number_of_trips = st.number_input("Number of Trips (avg per year)", min_value=1.0, max_value=22.0, value=3.0)
        passport = st.selectbox("Has Passport", ["No", "Yes"])
        pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
        own_car = st.selectbox("Owns a Car", ["No", "Yes"])
        number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0.0, max_value=3.0, value=0.0)
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Monthly Income", min_value=1000.0, max_value=98678.0, value=20000.0)
        duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=5.0, max_value=127.0, value=15.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": number_of_person_visiting,
        "NumberOfFollowups": number_of_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": number_of_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"This customer is likely to purchase the package. (confidence: {probability:.1%})")
    else:
        st.warning(f"This customer is unlikely to purchase the package. (confidence: {1 - probability:.1%})")

    with st.expander("View input data"):
        st.dataframe(input_data)
