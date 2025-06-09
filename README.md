# RTO (Return to Origin) Prediction System

A machine learning-based web application that predicts the likelihood of an order being returned to origin (RTO) in an e-commerce context. Built with Streamlit, XGBoost, and Python.

## Project Overview

This project implements a predictive system that helps e-commerce businesses identify potential RTO cases before shipping. It uses various order parameters to predict whether an order is likely to be returned to origin, helping businesses optimize their delivery operations and reduce losses.

### Features

- Real-time RTO prediction
- User-friendly web interface built with Streamlit
- Advanced machine learning model (XGBoost)
- Comprehensive data preprocessing
- Risk factor analysis and recommendations

### Input Parameters

The system accepts the following parameters:
- OrderId
- UserId
- OrderValue
- PaymentType (MPD/MPS/EMI/ADC/OPS)
- District
- OrderSource (android/desktop/mobile-site)
- OrderType (normal/express)
- DeliveryCharge
- OrderPlacedDay
- OrderConfirmDayOverPhone
- IsCartOrder (Yes/No)
- OrderFromPromotionalEvent (Yes/No)
- CourierService (Optional)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rto-prediction.git
cd rto-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Enter the order details in the web interface

4. Click "Predict RTO" to get the prediction result

## Project Structure

```
├── app.py                  # Streamlit web application
├── preprocess.py           # Data preprocessing module
├── models/
│   └── xgboost_model.json  # Trained XGBoost model
├── Dataset/
│   └── dataset.csv         # Training dataset
├── requirements.txt        # Project dependencies
├── LICENSE                # MIT License
└── README.md             # Project documentation
```

## Model Performance

The XGBoost model achieves:
- Accuracy: 84.56%
- ROC-AUC: 0.7303
- Precision: 0.3611
- Recall: 0.0455
- F1-Score: 0.0807

## Technologies Used

- Python 3.8+
- Streamlit
- XGBoost
- Pandas
- NumPy
- Scikit-learn

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Your Name

## Acknowledgments

- Dataset provided by [Your Source]
- Special thanks to contributors and maintainers