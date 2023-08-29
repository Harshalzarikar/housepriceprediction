from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Read the CSV file and preprocess the data
df1 = pd.read_csv(r"C:\Users\Asus\Desktop\vedash\Bengaluru_House_Data.csv")
df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
df3 = df2.dropna()


def is_float(x):
    try:
        float(x)
        return True
    except:
        return False


df3 = df3[df3['total_sqft'].apply(is_float)]
df3['total_sqft'] = df3['total_sqft'].apply(lambda x: float(x))

df3['price_per_sqft'] = df3['price'] * 100000 / df3['total_sqft']
df4 = df3.drop(['location', 'size'], axis='columns')

# Prepare input features (X) and target values (Y)
X = df4.drop('price', axis='columns')
Y = df4['price']

# Train the linear regression model
lr_clf = LinearRegression()
lr_clf.fit(X, Y)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    total_sqft = float(request.form['total_sqft'])
    bath = float(request.form['bath'])
    bhk = int(request.form['bhk'])

    # Create input feature array
    input_features = np.array([total_sqft, bath, bhk]).reshape(1, -1)

    # Make prediction using the trained model
    predicted_price = lr_clf.predict(input_features)[0]

    return render_template('index.html', prediction=predicted_price)


if __name__ == '__main__':
    app.run(debug=True)
