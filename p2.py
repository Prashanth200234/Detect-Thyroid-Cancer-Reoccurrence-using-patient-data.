import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("cleaned_dataset.csv")  # Update with actual dataset path

# Display initial dataset info
print("Dataset Overview:")
print(df.head())

# Encode categorical variables
label_encoders = {}
categorical_cols = ["Gender", "Smoking", "Hx Smoking", "Hx Radiotherapy",
                    "Thyroid Function", "Physical Examination", "Adenopathy",
                    "Pathology", "Focality", "Risk", "Response"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for future predictions

# Convert target variable 'Recurred' (Yes/No) to binary values
df["Recurred"] = df["Recurred"].map({"No": 0, "Yes": 1})

# Scale numerical columns
scaler = StandardScaler()
numerical_cols = ["Age", "T", "N", "M", "Stage"]
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Save label encoders & scaler
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Check correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()
# Define features and target
X = df.drop(columns=["Recurred"])  # Features
y = df["Recurred"]  # Target variable

# Split dataset into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
