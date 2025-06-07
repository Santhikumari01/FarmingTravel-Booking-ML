import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('modified_vehicle_booking_unique_owners.csv')
missing_values = data.isnull().sum()
print("Missing Values:")
print(missing_values)

data['vehicle_type_encoded'] = data['vehicle_type'].apply(lambda x: 1 if x == 'Farming' else 0)
data['availability_encoded'] = data['availability'].apply(lambda x: 1 if x == 'Yes' else 0)
X = data[['vehicle_type_encoded', 'cost']].copy()
y = data['availability_encoded']

scaler = StandardScaler()
X['cost'] = scaler.fit_transform(X[['cost']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print("Confusion Matrix:")
print(conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Available', 'Available'], yticklabels=['Not Available', 'Available'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC Curve")
plt.show()

def book_vehicle(vehicle_type, cost):
    vehicle_type_encoded = 1 if vehicle_type.lower() == 'farming' else 0
    features = pd.DataFrame([[vehicle_type_encoded, cost]], columns=['vehicle_type_encoded', 'cost'])
    features['cost'] = scaler.transform(features[['cost']])
    availability_prediction = model.predict(features)[0]
    if availability_prediction == 1:
        vehicle = data[(data['vehicle_type'] == vehicle_type.title()) & (data['cost'] == cost)]
        if not vehicle.empty:
            print("Vehicle booked successfully!")
            print(f"Vehicle: {vehicle.iloc[0]['vehicle_name']}, Owner: {vehicle.iloc[0]['owner_name']}, Contact: {vehicle.iloc[0]['contact_number']}")
        else:
            print("No matching vehicle found.")
    else:
        print("Vehicle not available for booking.")

book_vehicle("Farming", 1494)
book_vehicle("Traveling", 1821)
book_vehicle("Farming", 1702)