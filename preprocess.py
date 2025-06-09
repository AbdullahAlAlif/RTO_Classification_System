import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

def preprocess_input(input_data):
    # Convert input to DataFrame
    dt = pd.DataFrame([input_data])
    
    # Drop courierService as it was dropped in training
    dt = dt.drop(' courierService ', axis=1)
    
    # Calculate ConfirmationLatency in hours
    dt['OrderPlacedDay'] = pd.to_datetime(dt['OrderPlacedDay'])
    dt['OrderConfirmDayOverPhone'] = pd.to_datetime(dt['OrderConfirmDayOverPhone'])
    dt['ConfirmationLatency'] = (dt['OrderConfirmDayOverPhone'] - dt['OrderPlacedDay']).dt.total_seconds() / 3600
    dt['ConfirmationLatency'] = dt['ConfirmationLatency'].abs()
    
    # Drop OrderPlacedDay and OrderConfirmDayOverPhone
    dt = dt.drop(['OrderPlacedDay', 'OrderConfirmDayOverPhone'], axis=1)
    
    # Drop OrderId and UserId
    dt = dt.drop(['OrderId', 'UserId'], axis=1)
    
    # Convert boolean values to int
    boolean_cols = ['IsCartOrder', 'OrderFromPromotionalEvent']
    bool_map = {'Yes': np.int64(1), 'No': np.int64(0)}
    for col in boolean_cols:
        dt[col] = dt[col].map(bool_map)
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    numerical_cols = ['OrderValue', 'DeliveryCharge']
    dt[numerical_cols] = dt[numerical_cols].astype(np.float64)
    dt[numerical_cols] = scaler.fit_transform(dt[numerical_cols])
    
    # Create all required columns with zeros
    expected_columns = [
        'OrderValue', 'DeliveryCharge', 'IsCartOrder', 'OrderFromPromotionalEvent',
        'ConfirmationLatency', 'OrderType_express', 'OrderType_normal',
        'PaymentType_ADC', 'PaymentType_EMI', 'PaymentType_MPD', 'PaymentType_MPS',
        'PaymentType_OPS', 'District_Bagerhat', 'District_Bandarban', 'District_Barguna',
        'District_Barisal', 'District_Bhola', 'District_Bogra', 'District_Brahmanbaria',
        'District_Chandpur', 'District_Chapai-Nawabganj',
        'District_Chittagong (Outside City)', 'District_Chittagong City',
        'District_Chuadanga', 'District_Comilla', "District_Cox's-Bazar",
        'District_Dhaka', 'District_Dhaka (Outside City)', 'District_Dinajpur',
        'District_Faridpur', 'District_Feni', 'District_Gaibandha', 'District_Gazipur',
        'District_Gopalganj', 'District_Habiganj', 'District_Jamalpur',
        'District_Jessore', 'District_Jhalokati', 'District_Jhenaidah',
        'District_Joypurhat', 'District_Khagrachhari', 'District_Kishoreganj',
        'District_Kurigram', 'District_Kushtia', 'District_Lakshmipur',
        'District_Lalmonirhat', 'District_Madaripur', 'District_Magura',
        'District_Manikganj', 'District_Meherpur', 'District_Moulvibazar',
        'District_Munshiganj', 'District_Mymensingh', 'District_Naogaon',
        'District_Narail', 'District_Narayanganj', 'District_Narsingdi',
        'District_Natore', 'District_Netrokona', 'District_Nilphamari',
        'District_Noakhali', 'District_Pabna', 'District_Panchagarh',
        'District_Patuakhali', 'District_Pirojpur', 'District_Rajbari',
        'District_Rajshahi (Outside City)', 'District_Rajshahi City',
        'District_Rangamati', 'District_Rangpur', 'District_Satkhira',
        'District_Shariatpur', 'District_Sherpur', 'District_Sirajganj',
        'District_Sunamganj', 'District_Sylhet (Outside City)',
        'District_Sylhet City', 'District_Tangail', 'District_Thakurgaon',
        'District_khulna (Outside City)', 'District_khulna City',
        'OrderSource_android', 'OrderSource_desktop', 'OrderSource_mobile-site'
    ]
    
    # Initialize all columns with zeros
    for col in expected_columns:
        if col not in dt.columns:
            dt[col] = 0
    
    # Handle OrderType
    dt['OrderType_express'] = (dt['OrderType'] == 'express').astype(int)
    dt['OrderType_normal'] = (dt['OrderType'] == 'normal').astype(int)
    dt = dt.drop('OrderType', axis=1)
    
    # Handle PaymentType
    payment_type_cols = ['PaymentType_ADC', 'PaymentType_EMI', 'PaymentType_MPD', 
                        'PaymentType_MPS', 'PaymentType_OPS']
    for col in payment_type_cols:
        payment_type = col.replace('PaymentType_', '')
        dt[col] = (dt['PaymentType'] == payment_type).astype(int)
    dt = dt.drop('PaymentType', axis=1)
    
    # Handle District
    district_prefix = 'District_'
    current_district = dt['District'].iloc[0]
    district_col = f'{district_prefix}{current_district}'
    if district_col in expected_columns:
        dt[district_col] = 1
    dt = dt.drop('District', axis=1)
    
    # Handle OrderSource
    order_source_cols = ['OrderSource_android', 'OrderSource_desktop', 'OrderSource_mobile-site']
    for col in order_source_cols:
        source = col.replace('OrderSource_', '')
        dt[col] = (dt['OrderSource'] == source).astype(int)
    dt = dt.drop('OrderSource', axis=1)
    
    # Ensure columns are in the correct order
    dt = dt[expected_columns]
    
    return dt