import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import time
import threading
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request

# Flask 
app = Flask(__name__)

# Load and preprocess the dataset
data = pd.read_csv(r"dataset.csv", encoding='utf-8')
data.columns = data.columns.str.strip()  # Remove spaces

# EDA
print(data.info())
print(data.describe())
print(data.isnull().sum())

# figure with subplots
fig, ax = plt.subplots(figsize=(12, 8))
sns.countplot(ax=ax, x='Label', data=data, palette='viridis')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=12)
ax.set_title('Label Distribution', fontsize=16)
ax.set_xlabel('Label', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
plt.tight_layout()
plt.show()

# Data Prep
features = data.drop(columns=['Label'])
labels = data['Label'].map(lambda x: 1 if x != 'BENIGN' else 0)
features.replace([np.inf, -np.inf], np.nan, inplace=True)
features.dropna(inplace=True)
labels = labels[features.index]

scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(features_normalized, labels, test_size=0.2, random_state=42)

# LR Model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Evaluation
y_pred_logistic = logistic_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logistic))
print(classification_report(y_test, y_pred_logistic))

# DL Model
deep_model = Sequential()
deep_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
deep_model.add(Dense(32, activation='relu'))
deep_model.add(Dense(1, activation='sigmoid'))
deep_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
deep_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# DL Evaluation
y_pred_deep = (deep_model.predict(X_test) > 0.5).astype("int32")
print("Deep Learning Model Accuracy:", accuracy_score(y_test, y_pred_deep))
print(classification_report(y_test, y_pred_deep))

# Real-time detection func
def real_time_detection(new_data):
    new_data_normalized = scaler.transform(new_data)
    logistic_prediction = logistic_model.predict(new_data_normalized)
    deep_prediction = (deep_model.predict(new_data_normalized) > 0.5).astype("int32")
    if logistic_prediction == 0 and deep_prediction == 0:
        print("Normal Traffic")
    else:
        print("Intrusion Detected")

# Eg use with dummy real-time data
new_data_example = np.array([features.iloc[0]])
real_time_detection(new_data_example)

# Simulating data reading
def simulate_real_time_detection(data, interval=5):
    for i in range(len(data)):
        new_data = np.array([data.iloc[i]])
        real_time_detection(new_data)
        time.sleep(interval)  # Interval between checks

# Deploy conti.
def run_detection_service(data, interval=5):
    while True:
        simulate_real_time_detection(data, interval)

# Run detect in separate thread
detection_thread = threading.Thread(target=run_detection_service, args=(features, 5))
detection_thread.start()

# route for home page
@app.route('/')
def home():
    return render_template('index.html')

# route data sub
@app.route('/detect', methods=['POST'])
def detect_intrusion():
    data_file = request.files.get('data_file')
    real_time_data = request.form.get('real_time_data')

    if data_file:
        data = pd.read_csv(data_file)
        intrusion_result = perform_intrusion_detection(data, scaler, logistic_model, deep_model)
    elif real_time_data:
        real_time_data = pd.read_csv(pd.compat.StringIO(real_time_data))
        intrusion_result = perform_intrusion_detection(real_time_data, scaler, logistic_model, deep_model)
    else:
        return 'Error: No data provided'

    return render_template('result.html', intrusion_result=intrusion_result)

def perform_intrusion_detection(data, scaler, logistic_model, deep_model):
    features = data.drop(columns=['Label'])
    features_normalized = scaler.transform(features)
    logistic_prediction = logistic_model.predict(features_normalized)
    deep_prediction = (deep_model.predict(features_normalized) > 0.5).astype("int32")
    if logistic_prediction == 0 and deep_prediction == 0:
        return "Normal Traffic"
    else:
        return "Intrusion Detected"

if __name__ == '__main__':
    app.run(debug=True)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import time
import threading
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request

# Flask
app = Flask(__name__)

data = pd.read_csv(r"dataset.csv", encoding='utf-8')
data.columns = data.columns.str.strip()  

# EDA
print(data.info())
print(data.describe())
print(data.isnull().sum())

# figure with subplots
fig, ax = plt.subplots(figsize=(12, 8))
sns.countplot(ax=ax, x='Label', data=data, palette='viridis')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=12)
ax.set_title('Label Distribution', fontsize=16)
ax.set_xlabel('Label', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
plt.tight_layout()
plt.show()

features = data.drop(columns=['Label'])
labels = data['Label'].map(lambda x: 1 if x != 'BENIGN' else 0)
features.replace([np.inf, -np.inf], np.nan, inplace=True)
features.dropna(inplace=True)
labels = labels[features.index]

scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(features_normalized, labels, test_size=0.2, random_state=42)

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

y_pred_logistic = logistic_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logistic))
print(classification_report(y_test, y_pred_logistic))

# Model
deep_model = Sequential()
deep_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
deep_model.add(Dense(32, activation='relu'))
deep_model.add(Dense(1, activation='sigmoid'))
deep_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
deep_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Model Evaluation
y_pred_deep = (deep_model.predict(X_test) > 0.5).astype("int32")
print("Deep Learning Model Accuracy:", accuracy_score(y_test, y_pred_deep))
print(classification_report(y_test, y_pred_deep))

# Real-time Detect func
def real_time_detection(new_data):
    new_data_normalized = scaler.transform(new_data)
    logistic_prediction = logistic_model.predict(new_data_normalized)
    deep_prediction = (deep_model.predict(new_data_normalized) > 0.5).astype("int32")
    if logistic_prediction == 0 and deep_prediction == 0:
        print("Normal Traffic")
    else:
        print("Intrusion Detected")

# Eg usage 
new_data_example = np.array([features.iloc[0]])
real_time_detection(new_data_example)

# Simulate reading
def simulate_real_time_detection(data, interval=5):
    for i in range(len(data)):
        new_data = np.array([data.iloc[i]])
        real_time_detection(new_data)
        time.sleep(interval)  

# Deploy service
def run_detection_service(data, interval=5):
    while True:
        simulate_real_time_detection(data, interval)

# Run in separate thread
detection_thread = threading.Thread(target=run_detection_service, args=(features, 5))
detection_thread.start()

# route for home pg
@app.route('/')
def home():
    return render_template('index.html')

# route handle data sub
@app.route('/detect', methods=['POST'])
def detect_intrusion():
    data_file = request.files.get('data_file')
    real_time_data = request.form.get('real_time_data')

    if data_file:
        data = pd.read_csv(data_file)
        intrusion_result = perform_intrusion_detection(data, scaler, logistic_model, deep_model)
    elif real_time_data:
        real_time_data = pd.read_csv(pd.compat.StringIO(real_time_data))
        intrusion_result = perform_intrusion_detection(real_time_data, scaler, logistic_model, deep_model)
    else:
        return 'Error: No data provided'

    return render_template('result.html', intrusion_result=intrusion_result)

def perform_intrusion_detection(data, scaler, logistic_model, deep_model):
    features = data.drop(columns=['Label'])
    features_normalized = scaler.transform(features)
    logistic_prediction = logistic_model.predict(features_normalized)
    deep_prediction = (deep_model.predict(features_normalized) > 0.5).astype("int32")
    if logistic_prediction == 0 and deep_prediction == 0:
        return "Normal Traffic"
    else:
        return "Intrusion Detected"

if __name__ == '__main__':
    app.run(debug=True)
