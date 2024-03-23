# from flask import Flask, request, jsonify
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)
# # Load dataset
# df = pd.read_csv('Instagram forecast analysis.csv')

# # Convert 'date time' column to datetime
# df['date time'] = pd.to_datetime(df['Date'])

# # Extract features
# df['hour'] = df['date time'].dt.hour
# df['day_of_week'] = df['date time'].dt.dayofweek

# # Split dataset
# X = df[['hour', 'day_of_week']]
# y = df['Instagram reach']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model with Gradient Boosting regression
# model = GradientBoostingRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# @app.route('/evaluate', methods=['GET'])
# def evaluate_model():
#     # Evaluate model
#     y_pred = model.predict(X_test)
#     rmse = mean_squared_error(y_test, y_pred, squared=False)
#     return jsonify({'RMSE': rmse})

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     start_date = data['start_date']
#     end_date = data['end_date']
#     date_range = pd.date_range(start=start_date, end=end_date, freq='H')
#     predictions = model.predict(pd.DataFrame({'hour': date_range.hour, 'day_of_week': date_range.dayofweek}))
#     best_times = date_range[predictions.argmax()]
#     return jsonify({'best_times': best_times.strftime('%Y-%m-%d %H:%M:%S')})

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Read the dataset from the specified file path
df = pd.read_csv("Marketing campaign dataset.csv", low_memory=False)
df['creative_width'].fillna(0, inplace=True)
df['creative_height'].fillna(0, inplace=True)
df['template_id'].fillna(0, inplace=True)
df['approved_budget'].fillna(0, inplace=True)
df.drop(columns=['position_in_content','unique_reach','total_reach','max_bid_cpm'], inplace=True)
df['ctr'] = (df['clicks'] / df['impressions']) * 100

# Split the data into features (X) and target variable (y)
X = df[['clicks', 'impressions']]
y = df['ctr']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

@app.route('/predict_ctr', methods=['POST'])
def predict_ctr():
    data = request.get_json()
    clicks = data['clicks']
    impressions = data['impressions']
    X_pred = np.array([[clicks, impressions]])
    predicted_ctr = model.predict(X_pred)[0]
    return jsonify({'predicted_ctr': predicted_ctr})

@app.route('/best_campaign_date', methods=['GET'])
def best_campaign_date():
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')
    
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    date_range = pd.date_range(start=start_date, end=end_date)
    
    best_date = None
    best_ctr = -1
    for date in date_range:
        clicks = df['clicks'].mean()  # Use the average clicks for the prediction
        impressions = df['impressions'].mean()  # Use the average impressions for the prediction
        X_pred = np.array([[clicks, impressions]])
        predicted_ctr = model.predict(X_pred)[0]
        if predicted_ctr > best_ctr:
            best_ctr = predicted_ctr
            best_date = date.strftime('%Y-%m-%d')
    
    return jsonify({'best_campaign_date': best_date, 'expected_ctr': best_ctr})

if __name__ == '__main__':
    app.run(debug=True, port=6000)
