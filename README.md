# FarmingTravel-Booking-ML
This project introduces a Machine Learning-based Vehicle Booking System that supports reservations for both Traveling and Farming vehicles. While existing booking platforms mostly cater to travel vehicles like cars and bikes, this system addresses a critical gap by also providing a dedicated platform for Farming vehicle reservations such as tractors, harvesters, and seeders.

By using the powerful XGBoost algorithm, the system predicts whether a selected vehicle is available for booking, based on historical data and user input.

📌 Problem Statement
There are numerous online booking systems available for travel vehicles. However, no dedicated platform exists for reserving farming vehicles, which are essential for rural and agricultural users. This project aims to bridge that gap by building an intelligent system that caters to both categories.
🔍 Dataset Overview
A custom dataset was prepared with the following structure:
Column Name	Description
vehicle_type	Type of vehicle (Traveling / Farming)
vehicle_name	Name of the vehicle (e.g., Van, Tractor)
cost	Rental cost of the vehicle
owner_name	Owner's name or unique identifier
contact_number	Owner’s contact number
availability	Whether the vehicle is available (Yes / No)
53% Farming vehicles
47% Traveling vehicles
🧠 Machine Learning Approach
Model Used: XGBoost Classifier
Target Variable: availability
Categorical Encoding: One-Hot Encoding for vehicle_type and vehicle_name
Model Evaluation Metrics:
Accuracy
Precision
Recall
F1-score
Confusion Matrix
ROC Curve
The data is split into training and testing sets to validate the model’s performance.
🏆 Results
XGBoost Model Accuracy: 56.54%
Performance was superior compared to traditional models like Logistic Regression.
🎯 Features of the System
✅ User selects vehicle type and vehicle name
✅ System predicts availability
✅ If available:
Booking is confirmed
Displays vehicle name, owner’s name, and contact number
❌ If unavailable:
Displays notification indicating unavailability
📊 Includes visualizations:
Confusion matrix as a heatmap
ROC Curve for classification performance
