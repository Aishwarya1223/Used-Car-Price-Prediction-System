import streamlit as st
import joblib
import pandas as pd
import pickle
import category_encoders as ce

@st.cache_resource
def load_encoders():
    with open("picklefile_preprocessors/ohe_encoder.pkl", "rb") as f:
        ohe = pickle.load(f)
    with open("picklefile_preprocessors/target_encoder.pkl", "rb") as f:
        target_encoder = pickle.load(f)
    return ohe, target_encoder

def preprocess_input(df: pd.DataFrame, ohe, target_encoder) -> pd.DataFrame:
    ohe_cols = ['transmission', 'fuelType']
    ohe_encoded = ohe.transform(df[ohe_cols])
    ohe_encoded_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(ohe_cols), index=df.index)

    df_encoded = df.drop(columns=ohe_cols).reset_index(drop=True)
    df_encoded = pd.concat([df_encoded, ohe_encoded_df.reset_index(drop=True)], axis=1)

    target_encoded_df = target_encoder.transform(df[['model', 'brand']].copy())
    df_encoded = pd.concat([df_encoded, target_encoded_df.reset_index(drop=True)], axis=1)

    df_encoded.drop(columns=['model', 'brand'], inplace=True)
    return df_encoded


# --- Streamlit Page Config ---
st.set_page_config(page_title="Used Car Price Prediction", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "⚙️ Settings", "ℹ️ About"])

# --- Car Brand & Model Mapping ---
brand_model_map = {
    'Audi': ['A1', 'A3', 'A4', 'A6'],
    'BMW': ['1 Series', '3 Series', '5 Series'],
    'cclass': ['C180', 'C200'],
    'Focus': ['Zetec', 'Titanium'],
    'Ford': ['Fiesta', 'Focus', 'Kuga'],
    'Hyundi': ['i10', 'i20', 'i30'],
    'Merc': ['C-Class', 'E-Class'],
    'Skoda': ['Fabia', 'Octavia', 'Superb'],
    'toyota': ['Aygo', 'Yaris', 'Corolla'],
    'Vauxhall': ['Corsa', 'Astra'],
    'VW': ['Golf', 'Passat', 'Polo']
}
st.markdown(
        """
        <h1 style='text-align: center;font-size:30px'>🚗 USED CAR PRICE PREDICTION APP</h1>
        <p style='text-align: center;'>This app predicts the price of a used car using a machine learning model.</p>
        """,
        unsafe_allow_html=True
    )
# --- Page: Home ---
if page == "🏠 Home":
    

    # --- Input Form Layout ---
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Brand", sorted(brand_model_map.keys()))
        model = st.selectbox("Model", brand_model_map.get(brand, []))
        year = st.number_input("Year", 1990, 2025, 2018)
        mileage = st.number_input("Mileage (in km)", 0, 300000, 50000)
        engine_size = st.number_input("Engine Size (in Liters)", 0.5, 6.0, 1.4, step=0.1)

    with col2:
        transmission = st.selectbox("Transmission", ["Manual", "Automatic", "Semi-Auto", "Other"])
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric"])
        tax = st.number_input("Tax (£)", 0, 600, 150)
        mpg = st.number_input("Miles Per Gallon", 10.0, 120.0, 55.4, step=0.1)

    # --- Predict Button ---
    if st.button('🔍 Predict Price'):
        input_data = {
            "brand": brand,
            "model": model,
            "year": year,
            "mileage": mileage,
            "engineSize": engine_size,
            "transmission": transmission,
            "fuelType": fuel_type,
            "tax": tax,
            "mpg": mpg
        }

        input_df = pd.DataFrame([input_data])

        # --- Load the Best Model ---
        
        model = joblib.load("best_model/best_model.pkl")  # or h2o.load_model()
        ohe, target_encoder = load_encoders()
        input_df = preprocess_input(input_df, ohe, target_encoder)

        prediction = model.predict(input_df)[0]

        st.markdown("---")
        st.subheader(f"💰 Estimated Price: **£{prediction:,.2f}**")

# --- Page: Settings ---
elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    st.markdown("You can configure model preferences or application settings here.")
    # Optional: Add checkboxes/sliders to configure prediction mode, thresholds, etc.

# --- Page: About ---
elif page == "ℹ️ About":
    st.markdown(
        """
        <h1 style='text-align: center;'>ℹ️ About</h1>
        <p style='text-align: center;'>This app was built by <b>Aishwarya R</b> as a portfolio project.</p>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""
    ### 🔧 Features:
    - Real-time car price prediction  
    - Auto model selection via CI/CD  
    - Dockerized with Streamlit UI  
    - Retrains weekly using GitHub Actions

    🌐 GitHub: [Used Car Price Prediction System](https://github.com/Aishwarya1223/Used-Car-Price-Prediction-System)
    """)

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center;'>Made by <b>Aishwarya R</b></p>", unsafe_allow_html=True)
