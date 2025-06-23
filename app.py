import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
from preprocess import preprocess_input
import joblib

# Set page title
st.title('RTO Prediction System')

# Load the Random Forest model
model = joblib.load('models/random_forest_model.pkl')

# Create input form
st.header('Enter Order Details')

col1, col2 = st.columns(2)

with col1:
    order_value = st.number_input('Order Value', min_value=0.0)
    payment_type = st.selectbox('Payment Type', ['ADC', 'EMI', 'MPD', 'MPS', 'OPS'])
    district = st.selectbox('District', [
        'Bagerhat', 'Bandarban', 'Barguna', 'Barisal', 'Bhola', 'Bogra', 
        'Brahmanbaria', 'Chandpur', 'Chapai-Nawabganj', 'Chittagong (Outside City)',
        'Chittagong City', 'Chuadanga', 'Comilla', "Cox's-Bazar", 'Dhaka',
        'Dhaka (Outside City)', 'Dinajpur', 'Faridpur', 'Feni', 'Gaibandha',
        'Gazipur', 'Gopalganj', 'Habiganj', 'Jamalpur', 'Jessore', 'Jhalokati',
        'Jhenaidah', 'Joypurhat', 'Khagrachhari', 'Kishoreganj', 'Kurigram',
        'Kushtia', 'Lakshmipur', 'Lalmonirhat', 'Madaripur', 'Magura',
        'Manikganj', 'Meherpur', 'Moulvibazar', 'Munshiganj', 'Mymensingh',
        'Naogaon', 'Narail', 'Narayanganj', 'Narsingdi', 'Natore', 'Netrokona',
        'Nilphamari', 'Noakhali', 'Pabna', 'Panchagarh', 'Patuakhali',
        'Pirojpur', 'Rajbari', 'Rajshahi (Outside City)', 'Rajshahi City',
        'Rangamati', 'Rangpur', 'Satkhira', 'Shariatpur', 'Sherpur',
        'Sirajganj', 'Sunamganj', 'Sylhet (Outside City)', 'Sylhet City',
        'Tangail', 'Thakurgaon', 'khulna (Outside City)', 'khulna City'
    ])
    # Add time input for OrderPlacedDay
    order_placed = st.date_input('Order Placed Day')
    order_placed_time = st.time_input('Order Placed Time', value=None, key='order_placed_time')

with col2:
    order_source = st.selectbox('Order Source', ['android', 'desktop', 'mobile-site'])
    order_type = st.selectbox('Order Type', ['express', 'normal'])
    delivery_charge = st.number_input('Delivery Charge', min_value=0.0)
    order_confirm = st.date_input('Order Confirmation Day')
    order_confirm_time = st.time_input('Order Confirmation Time', value=None, key='order_confirm_time')
    is_cart_order = st.selectbox('Is Cart Order', ['Yes', 'No'])
    is_promotional = st.selectbox('Is Promotional Order', ['Yes', 'No'])

# Combine date and time for order_placed and order_confirm
def combine_date_time(date_obj, time_obj):
    if time_obj is None:
        time_obj = time(0, 0)
    return datetime.combine(date_obj, time_obj)

# Create prediction button
if st.button('Predict RTO'):
    # Validate dates
    order_placed_dt = combine_date_time(order_placed, order_placed_time)
    order_confirm_dt = combine_date_time(order_confirm, order_confirm_time)
    if order_confirm_dt < order_placed_dt:
        st.error("Error: Order Confirmation Date/Time cannot be earlier than Order Placed Date/Time!")
    else:
        # Prepare input data (removed OrderId, UserId, and courierService)
        input_data = {
            'OrderValue': order_value,
            'PaymentType': payment_type,
            'District': district,
            'OrderSource': order_source,
            'OrderType': order_type,
            'DeliveryCharge': delivery_charge,
            'OrderPlacedDay': order_placed_dt,
            'OrderConfirmDayOverPhone': order_confirm_dt,
            'IsCartOrder': is_cart_order,
            'OrderFromPromotionalEvent': is_promotional
        }
    try:
        # Preprocess the input
        processed_data = preprocess_input(input_data)
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0]
        # Display results
        st.header('Prediction Results')
        if prediction == 1:
            st.error('⚠️ High Risk of RTO')
            st.write(f'Probability of RTO: {probability[1]:.2%}')
            # Display risk factors
            st.subheader('Risk Factors:')
            risk_factors = []
            if district not in ['Dhaka', 'Chittagong City', 'Gazipur']:
                risk_factors.append('- Order from a high-risk district')
            if payment_type == 'MPD':
                risk_factors.append('- Mobile payment on delivery has higher RTO risk')
            if float(delivery_charge) > 100:
                risk_factors.append('- High delivery charge may increase RTO risk')
            if (order_confirm_dt - order_placed_dt).days > 2:
                risk_factors.append('- Long confirmation delay increases RTO risk')
            for factor in risk_factors:
                st.write(factor)
            st.subheader('Recommendations:')
            st.write('1. Verify customer contact information')
            st.write('2. Consider alternative payment methods')
            st.write('3. Ensure proper customer communication')
            st.write('4. Verify delivery address thoroughly')
        else:
            st.success('✅ Low Risk of RTO')
            st.write(f'Probability of RTO: {probability[1]:.2%}')
    except Exception as e:
        st.error(f'Error occurred: {str(e)}')
