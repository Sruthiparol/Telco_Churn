# streamlitwebapp.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# Import the custom class from the new utility file
from utils import TargetEncoder # <-- CRITICAL FIX

# -------------------------- CONFIG --------------------------
DATA_PATH = "beer-servings.csv" 
MODELS_DIR = Path("models")
RANDOM_STATE = 42
# ------------------------------------------------------------

st.set_page_config(
    page_title="🍺 Alcohol Consumption Predictor", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# NOTE: The TargetEncoder class definition has been removed from here
# and moved to utils.py to prevent Joblib loading errors.


# -------------------------- DATA LOADING --------------------------

@st.cache_data(show_spinner="Loading and preparing data...")
def load_and_preprocess_data():
    """Loads and cleans the dataset."""
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"FATAL: Data file not found at: {DATA_PATH}. Ensure the CSV is in the root directory.")
        return pd.DataFrame()
    
    df['continent'] = df['continent'].fillna('Unknown')
    df = df.dropna()
    return df

# -------------------------- MODEL LOADING --------------------------

@st.cache_resource(show_spinner="Loading saved model components...")
def load_saved_components():
    """
    Loads all saved components (model, encoders, scaler) from the 'models' directory.
    """
    components = {}
    try:
        # Load the dictionary saved by test_train.py
        model_data = joblib.load(MODELS_DIR / 'best_model.joblib')
        components['model'] = model_data['model']
        # Accessing the required keys saved by the training script
        components['model_name'] = model_data['name'] 
        components['r2_score'] = model_data.get('r2_test', 'N/A') 
        
        # Load preprocessors
        components['country_encoder'] = joblib.load(MODELS_DIR / 'country_encoder.joblib')
        components['continent_ohe'] = joblib.load(MODELS_DIR / 'cont_ohe.joblib')
        components['scaler'] = joblib.load(MODELS_DIR / 'scaler.joblib')
        return components
    except Exception as e:
        st.error(f"CRITICAL ERROR: Failed to load all model components from {MODELS_DIR}.\n\n**Action Required:** Ensure you ran **python test_train.py** locally and uploaded the resulting **models/** folder to GitHub.\n\nError: {e}")
        return None

# -------------------------- PREDICTION FUNCTION --------------------------

def predict_alcohol_litres(input_data: dict, components: dict) -> float:
    """Preprocesses input data and makes a prediction."""
    
    input_df = pd.DataFrame([input_data])
    
    # 1. Target Encode Country
    ce = components['country_encoder']
    input_df['country_encoded'] = ce.transform(input_df['country'])
    
    # 2. One-Hot Encode Continent
    ohe = components['continent_ohe']
    continent_ohe = ohe.transform(input_df[['continent']])
    continent_cols = [f'cont_{c}' for c in ohe.categories_[0]]
    continent_df = pd.DataFrame(continent_ohe, columns=continent_cols, index=input_df.index)
    
    # 3. Select features for scaling
    feature_data = input_df[['beer_servings', 'spirit_servings', 'wine_servings', 'country_encoded']]
    
    # 4. Scale features
    scaler = components['scaler']
    scaled_features = scaler.transform(feature_data)
    scaled_df = pd.DataFrame(scaled_features, columns=feature_data.columns, index=input_df.index)
    
    # 5. Final Input DataFrame
    final_input_df = pd.concat([scaled_df, continent_df], axis=1)

    # 6. Predict
    prediction = components['model'].predict(final_input_df.values)[0] 
    return prediction

# ========================== MAIN APP LOGIC ==========================

def main():
    
    df = load_and_preprocess_data()
    if df.empty:
        st.stop()

    components = load_saved_components()
    if not components:
        st.stop()

    # ------------------ Tabs ------------------
    tab_eda, tab_pred = st.tabs(["Data Insights & Charts", "Make a Prediction"])

    with tab_eda:
        st.markdown("## 📊 Exploratory Data Analysis (Infographics)")
        df_agg = df.groupby('continent')[['beer_servings', 'total_litres_of_pure_alcohol']].mean().reset_index()

        col_left, col_right = st.columns(2)

        with col_left:
            fig1 = px.bar(
                df_agg, 
                x='continent', 
                y='total_litres_of_pure_alcohol', 
                title='Avg. Total Pure Alcohol by Continent',
                color='continent',
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col_right:
            fig2 = px.scatter(
                df, 
                x='beer_servings', 
                y='total_litres_of_pure_alcohol', 
                color='continent',
                hover_data=['country'],
                title='Beer Servings vs. Total Pure Alcohol (by Country)',
            )
            st.plotly_chart(fig2, use_container_width=True)
            
    # ------------------ Prediction Section ------------------
    with tab_pred:
        st.markdown("## 🔮 Predict Total Litres of Pure Alcohol")
        st.info(f"Using **{components['model_name']}** (Test R²: **{components['r2_score']:.3f}**)")
        
        all_countries = sorted(df['country'].unique())
        all_continents = sorted(df['continent'].unique())

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            input_country = st.selectbox("Country", options=all_countries, index=all_countries.index('Germany') if 'Germany' in all_countries else 0)

        with col2:
            input_continent = st.selectbox("Continent", options=all_continents, index=all_continents.index('Europe') if 'Europe' in all_continents else 0)
            
        with col3:
            input_beer = st.number_input("Beer Servings", min_value=0, max_value=500, value=150, step=10)

        with col4:
            input_spirit = st.number_input("Spirit Servings", min_value=0, max_value=500, value=100, step=10)

        with col5:
            input_wine = st.number_input("Wine Servings", min_value=0, max_value=500, value=50, step=5)
        
        st.markdown("---")
        
        if st.button("Calculate Prediction", type="primary"):
            input_data = {
                'country': input_country,
                'beer_servings': input_beer,
                'spirit_servings': input_spirit,
                'wine_servings': input_wine,
                'continent': input_continent
            }
            
            with st.spinner('Calculating prediction...'):
                try:
                    result = predict_alcohol_litres(input_data, components)
                    st.success(f"### Predicted Total Litres of Pure Alcohol: **{result:.4f} Litres**")
                    st.caption("Prediction incorporates complex features: Target Encoding for country-level averages and One-Hot Encoding for continental influence.")
                except Exception as e:
                    st.error(f"An error occurred during prediction. Error: {e}")


if __name__ == '__main__':
    main()