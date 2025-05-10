import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
sc=StandardScaler()

#Loading scaleres and model
def load_resources():   
    model = joblib.load('randomforest_model.pkl')
    category_encoder = joblib.load('encoders/le_merchant_category.pkl')
    final_scaler = joblib.load("final_scaler.pkl")

    return model, category_encoder, final_scaler


def main():
    #title and header
    st.title('Credit Card Fraud Detection App')
    st.header('Enter transaction details:')

    # Load resources
    model, category_encoder,final_scaler= load_resources()

    # Take user input
    input_data = {}

    input_data['Transaction_amount'] = st.number_input('Transaction Amount ($)', min_value=0.0)
    input_data['trans_hour'] = st.slider('Transaction Hour (0–23)', 0, 23, 12)
    input_data['Merchant_category'] =st.selectbox('Transaction Category', [
    'gas_transport','grocery_pos','home','shopping_pos','kids_pets',
    'shopping_net','entertainment','food_dining','personal_care',
    'health_fitness','misc_pos','misc_net','grocery_net','travel'
])
    input_data['zip'] = st.number_input('Zip Code', min_value=-100, max_value=100000, value=0)
    
    
    # Create a DataFrame with user input
    input_df = pd.DataFrame([input_data])
    
    input_df['Merchant_category'] = category_encoder.transform([input_df['Merchant_category'][0]])

    # 2. Apply log transformation to 'Transaction_amount'
    input_df['Transaction_amount'] = np.log(input_df['Transaction_amount'][0] + 1)

    # 3. Apply individual scaling to each column
    input_df= final_scaler.transform(input_df)

    
    # Prediction
    if st.button('Predict'):
        expected_columns = ['Transaction_amount', 'trans_hour', 'Merchant_category', 'zip']
        prediction = model.predict(input_df)
        
        if prediction[0] == 1:
            st.error('Fraudulent Transaction Detected!')
            st.header('Don\'t worry, we are here to help you!')
            st.success('We successfully identified and stopped the suspicious transaction before it was completed — your money is safe.')
        else:
            st.success('✅ Legitimate Transaction')
            st.balloons()
# Run the app
if __name__ == '__main__':
    main()
