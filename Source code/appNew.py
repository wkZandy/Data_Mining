import streamlit as st
import pickle
import numpy as np
import pandas as pd
model=pickle.load(open('modelNew.pkl','rb'))

def main():
    st.title("Web")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">XGB prediction web</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    DEP_TIME_BLK = st.selectbox("DEP_TIME_BLK", ['0500-0559' ,'0600-0659', '0700-0759', '0800-0859', '0900-0959', '1000-1059', '1100-1159', '1200-1259', '1300-1359', '1400-1459', '1500-1559', '1600-1659', '1700-1759', '1800-1859', '1900-1959','2000-2059', '2100-2159', '2200-2259', '2300-2359'])
    CARRIER_NAME = st.selectbox("CARRIER_NAME", [
                                                'Allegiant Air',
                                                'American Airlines Inc.',
                                                'American Eagle Airlines Inc.',
                                                'Atlantic Southeast Airlines',
                                                'Comair Inc.',
                                                'Delta Air Lines Inc.',
                                                'Endeavor Air Inc.',
                                                'Frontier Airlines Inc.',
                                                'Hawaiian Airlines Inc.',
                                                'JetBlue Airways',
                                                'Mesa Airlines Inc.',
                                                'Midwest Airline, Inc.',
                                                'SkyWest Airlines Inc.',
                                                'Southwest Airlines Co.',
                                                'Spirit Air Lines',
                                                'United Air Lines Inc.'])
    
    DEPARTING_STATE = st.selectbox("DEPARTING_STATE", ['AL',
                                                        'AR',
                                                        'AZ',
                                                        'CA',
                                                        'CO',
                                                        'CT',
                                                        'DC',
                                                        'FL',
                                                        'GA',
                                                        'HI',
                                                        'IA',
                                                        'ID',
                                                        'IL',
                                                        'IN',
                                                        'KY',
                                                        'LA',
                                                        'MA',
                                                        'MD',
                                                        'ME',
                                                        'MI',
                                                        'MN',
                                                        'MO',
                                                        'NC',
                                                        'NE',
                                                        'NJ',
                                                        'NM',
                                                        'NV',
                                                        'NY',
                                                        'OH',
                                                        'OK',
                                                        'OR',
                                                        'PA',
                                                        'PR',
                                                        'RI',
                                                        'SC',
                                                        'TN',
                                                        'TX',
                                                        'UT',
                                                        'VA',
                                                        'WA',
                                                        'WI'])
    SEGMENT_NUMBER = st.number_input("SEGMENT_NUMBER", placeholder="Type a number...")
    PRCP     = st.number_input("PRCP", placeholder="Type a number...")
    LONGITUDE     = st.number_input("LONGITUDE", placeholder="Type a number...")

    safe_html="""  
      <div style="background-color:black;padding:10px >
       <h2 style="color:white;text-align:center;"> Your flight is likely to be delayed</h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:black;padding:10px >
       <h2 style="color:white ;text-align:center;"> Your flight will likely be on time </h2>
       </div>
    """
    
    # When the user clicks the submit button
    if st.button("Predict"):
        # Create a DataFrame from the input data
        data = {
            "DEP_TIME_BLK": [DEP_TIME_BLK],
            "CARRIER_NAME": [CARRIER_NAME],
            "DEPARTING_STATE": [DEPARTING_STATE],
            "SEGMENT_NUMBER": [SEGMENT_NUMBER],
            "PRCP": [PRCP],
            "LONGITUDE": [LONGITUDE],
        }
        df = pd.DataFrame(data)

        # Generate dummy variables for DEP_TIME_BLK
        df = pd.get_dummies(df, columns=["DEP_TIME_BLK",'CARRIER_NAME','DEPARTING_STATE'])
        
        # Ensure the DataFrame has all the dummy columns the model expects
        expected_cols = ["SEGMENT_NUMBER","PRCP", "LONGITUDE", 
                         "DEP_TIME_BLK_0600-0659",	"DEP_TIME_BLK_0700-0759", "DEP_TIME_BLK_0800-0859", "DEP_TIME_BLK_0900-0959",	"DEP_TIME_BLK_1000-1059", "DEP_TIME_BLK_1100-1159",	"DEP_TIME_BLK_1200-1259", "DEP_TIME_BLK_1300-1359",	"DEP_TIME_BLK_1400-1459", "DEP_TIME_BLK_1500-1559",	"DEP_TIME_BLK_1600-1659", "DEP_TIME_BLK_1700-1759",	"DEP_TIME_BLK_1800-1859", "DEP_TIME_BLK_1900-1959","DEP_TIME_BLK_2000-2059", "DEP_TIME_BLK_2100-2159", "DEP_TIME_BLK_2200-2259", "DEP_TIME_BLK_2300-2359", 
    "CARRIER_NAME_Allegiant Air",
    "CARRIER_NAME_American Airlines Inc.",
    "CARRIER_NAME_American Eagle Airlines Inc.",
    "CARRIER_NAME_Atlantic Southeast Airlines",
    "CARRIER_NAME_Comair Inc.",
    "CARRIER_NAME_Delta Air Lines Inc.",
    "CARRIER_NAME_Endeavor Air Inc.",
    "CARRIER_NAME_Frontier Airlines Inc.",
    "CARRIER_NAME_Hawaiian Airlines Inc.",
    "CARRIER_NAME_JetBlue Airways",
    "CARRIER_NAME_Mesa Airlines Inc.",
    "CARRIER_NAME_Midwest Airline, Inc.",
    "CARRIER_NAME_SkyWest Airlines Inc.",
    "CARRIER_NAME_Southwest Airlines Co.",
    "CARRIER_NAME_Spirit Air Lines",
    "CARRIER_NAME_United Air Lines Inc.",
    "DEPARTING_STATE_AL",
    "DEPARTING_STATE_AR",
    "DEPARTING_STATE_AZ",
    "DEPARTING_STATE_CA",
    "DEPARTING_STATE_CO",
    "DEPARTING_STATE_CT",
    "DEPARTING_STATE_DC",
    "DEPARTING_STATE_FL",
    "DEPARTING_STATE_GA",
    "DEPARTING_STATE_HI",
    "DEPARTING_STATE_IA",
    "DEPARTING_STATE_ID",
    "DEPARTING_STATE_IL",
    "DEPARTING_STATE_IN",
    "DEPARTING_STATE_KY",
    "DEPARTING_STATE_LA",
    "DEPARTING_STATE_MA",
    "DEPARTING_STATE_MD",
    "DEPARTING_STATE_ME",
    "DEPARTING_STATE_MI",
    "DEPARTING_STATE_MN",
    "DEPARTING_STATE_MO",
    "DEPARTING_STATE_NC",
    "DEPARTING_STATE_NE",
    "DEPARTING_STATE_NJ",
    "DEPARTING_STATE_NM",
    "DEPARTING_STATE_NV",
    "DEPARTING_STATE_NY",
    "DEPARTING_STATE_OH",
    "DEPARTING_STATE_OK",
    "DEPARTING_STATE_OR",
    "DEPARTING_STATE_PA",
    "DEPARTING_STATE_PR",
    "DEPARTING_STATE_RI",
    "DEPARTING_STATE_SC",
    "DEPARTING_STATE_TN",
    "DEPARTING_STATE_TX",
    "DEPARTING_STATE_UT",
    "DEPARTING_STATE_VA",
    "DEPARTING_STATE_WA",
    "DEPARTING_STATE_WI"]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0

        # Reorder columns to match the training data
        df = df[expected_cols]


        # Convert DataFrame to NumPy array
        input_data = df.to_numpy()

        # Make prediction
        prediction = model.predict(input_data)
        if prediction == 0:
            st.markdown(danger_html,unsafe_allow_html=True)
        else:
            st.markdown(safe_html,unsafe_allow_html=True)
        st.success(f'The prediction is: {prediction}')

if __name__=='__main__':
    main()
