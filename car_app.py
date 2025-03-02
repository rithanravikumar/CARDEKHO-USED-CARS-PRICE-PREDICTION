import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64
from sklearn.preprocessing import OrdinalEncoder


@st.cache_resource
# Load models
def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)
    
def set_background_image_local(image_path):
    with open(image_path, "rb") as file:
        data = file.read()
    base64_image = base64.b64encode(data).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{base64_image}");
            background-size: contain;
            background-position: fit;
            background-repeat: repeat;
            background-attachment: fixed;
        }}     
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_image_local("D:/USED CARS/greenbg.jpg")

model_car=load_model("D:/USED CARS/carmodel_final.pkl")

encoder_city=load_model("D:/USED CARS/encoder_city.pkl")
encoder_Insurance_Validity=load_model("D:/USED CARS/encoder_Insurance_Validity (1).pkl")
encoder_bt=load_model("D:/USED CARS/encoder_bt.pkl")
encoder_ft=load_model("D:/USED CARS/encoder_ft.pkl")
encoder_oem=load_model("D:/USED CARS/encoder_oem.pkl")
encoder_model=load_model("D:/USED CARS/encoder_model.pkl")
encoder_transmission=load_model("D:/USED CARS/encoder_transmission.pkl")
encoder_variantName=load_model("D:/USED CARS/encoder_variantName.pkl")

ml_df=pd.read_excel("D:/USED CARS/ml_dl.xlsx")
st.title("")
st.title("Car Price Prediction App")

categorical_features = ["city", "ft", "bt", "transmission", "oem", "model", "variantName", "Insurance Validity"]
dropdown_options = {feature: ml_df[feature].unique().tolist() for feature in categorical_features}

tab1, tab2 = st.tabs(["Home", "Predict"])
with tab1:
    st.markdown("""
                **1. Introduction:**
                In today’s fast-paced automotive industry, accurately pricing a vehicle is essential for both buyers and sellers.
                The Car Price Prediction App provides a smart solution to estimate car prices using machine learning.
                By analyzing historical data and applying predictive analytics, this tool empowers users to make well-informed decisions.


                **2. Problem Statement:**
                Determining a car’s fair market value involves assessing multiple factors such as brand, model, manufacturing year, 
                mileage, fuel type, and transmission. Manually evaluating these aspects can be tedious and complex. 
                The Car Price Prediction App streamlines this process by delivering quick and precise price estimates.

            
                **3. Key Features:**
                Intuitive Interface :- A user-friendly and interactive UI built with Streamlit.

                Machine Learning Model :- Employs an advanced regression algorithm (XGBRegressor) trained on a comprehensive dataset.

                Customizable Inputs :- Users can specify car details, including brand, model, year, fuel type, and transmission.

                Instant Price Estimation :- Provides real-time predictions based on user inputs.

                Comparison Feature :- Enables users to compare multiple cars for better decision-making.
                
                **4. Target Audience:**
                Car Buyers & Sellers :- Individuals seeking a fair market price for used cars.

                Dealerships & Businesses :- Car dealerships and resellers needing quick car price evaluations.

                Financial Institutions :- Banks and insurance companies assessing car values for loans and policies.
                
                **5. Technologies Used:**
                Frontend:- Built with Streamlit for an interactive web experience.

                Backend:- Developed using Python and key machine learning libraries like Scikit-learn, XGBoost, and Pandas.

                Model Deployment:- The trained model is seamlessly integrated into the Streamlit app for real-time predictions.
                
                **6. MLFlow:**
                The project incorporates MLflow to efficiently track and manage machine learning experiments within the Streamlit application.
                It logs essential details such as model parameters, performance metrics (MSE, MAE, R²), and trained models. 
                MLflow Link: http://127.0.0.1:5000/
                
                **7. Conclusion:**
                The Car Price Prediction App serves as a valuable tool for individuals and businesses looking to determine car prices with ease. 
                By leveraging machine learning, it enhances transparency and efficiency in the buying and selling process, ensuring informed decision-making.
                """)
with tab2:
    a1,a2,a3=st.columns(3)
    a4,a5,a6=st.columns(3)
    a7,a8,a9=st.columns(3)
    a10,a11,a12=st.columns(3)
    a13,a14=st.columns(2)
    
    with a1:
        city_select=st.selectbox("Select City",dropdown_options["city"])
        city=encoder_city.transform([[city_select]])[0][0]
    with a2:
        ft_select=st.selectbox("Select fuel Type",dropdown_options["ft"])
        ft=encoder_ft.transform([[ft_select]])[0][0]
    with a3:
        bt_select=st.selectbox("Select Body Type",dropdown_options["bt"])
        bt=encoder_bt.transform([[bt_select]])[0][0]
    with a4:
        km=st.number_input("Enter KM driven",min_value=10)
    with a5:
        transmission_select=st.selectbox("Select Transmission",dropdown_options["transmission"])
        transmission=encoder_transmission.transform([[transmission_select]])[0][0]
    with a6:
        ownerNo=st.number_input("Enter no. of Owner's",min_value=1)
    with a7:
        oem_list=ml_df[ml_df["ft"]==ft_select]["oem"]
        oem_filtered=oem_list.unique().tolist()
        oem_select=st.selectbox("Select car manufacture name",oem_filtered)
        oem=encoder_oem.transform([[oem_select]])[0][0]
    with a8:
        model_list=ml_df[ml_df["oem"]==oem_select]["model"]
        model_filtered=model_list.unique().tolist()
        model_select=st.selectbox("Select car Model name",model_filtered)
        model=encoder_model.transform([[model_select]])[0][0]
    with a9:
        modelYear=st.number_input("Enter car manufacture year",min_value=1900)
    with a10:
        variantName_list=ml_df[ml_df["model"]==model_select]["variantName"]
        variantName_filtered=variantName_list.unique().tolist()
        variantName_select=st.selectbox("Select Model variant Name",variantName_filtered)
        variantName=encoder_variantName.transform([[variantName_select]])[0][0]
    with a11:
        Registration_Year=st.number_input("Enter car registration year",min_value=1900)
    with a12:
        InsuranceValidity_select=st.selectbox("Select Insurance Type",dropdown_options["Insurance Validity"])
        InsuranceValidity=encoder_Insurance_Validity.transform([[InsuranceValidity_select]])[0][0]
    with a13:
        Seats=st.number_input("Enter seat capacity",min_value=4)
    with a14:
        EngineDisplacement=st.number_input("Enter Engine CC",min_value=799)
        
    if st.button('Predict'):
        input_data = pd.DataFrame([city,ft,bt,km,transmission,ownerNo,oem,model,modelYear,variantName,Registration_Year,InsuranceValidity,Seats,EngineDisplacement])

        prediction = model_car.predict(input_data.values.reshape(1, -1))
                
        st.subheader("Predicted Car Price")
        st.markdown(f"### :blue[₹ {prediction[0]:,.2f}]")
