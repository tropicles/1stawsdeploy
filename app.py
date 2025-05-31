from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load and fit scaler on training data
# Assumes 'notebook/clean.csv' has your training features
train_df = pd.read_csv('notebook/clean.csv')
feature_cols = ['MedInc','HouseAge','AveRooms','AveBedrms',
                'Population','AveOccup','Latitude','Longitude']
X_train = train_df[feature_cols]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)

# Load trained regression model
with open('model/regression.pkl', 'rb') as f:
    regression = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predicted', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        # Collect and cast input features in correct order
        features = [
            float(request.form['MedInc']),
            float(request.form['HouseAge']),
            float(request.form['AveRooms']),
            float(request.form['AveBedrms']),
            float(request.form['Population']),
            float(request.form['AveOccup']),
            float(request.form['Latitude']),
            float(request.form['Longitude'])
        ]
        # Wrap into DataFrame for proper scaling
        df_new = pd.DataFrame([features], columns=feature_cols)
        # Scale features
        X_scaled = scaler.transform(df_new)
        # Predict
        y_pred = regression.predict(X_scaled)[0]
        # Render result
        return render_template('result.html', prediction=round(y_pred, 3))
    else:
        return render_template('result.html', prediction=0)

if __name__ == '__main__':
    app.run(debug=True)
