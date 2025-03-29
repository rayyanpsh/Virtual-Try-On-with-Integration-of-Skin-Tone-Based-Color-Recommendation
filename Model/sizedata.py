from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
# Load the dataset
df = pd.read_csv("D:/VTON/Model/size_dataset.csv")

# Display the first few rows


# Initialize label encoder
label_encoder = LabelEncoder()

# Encode the 'Size' column
from sklearn.preprocessing import LabelEncoder

# Initialize label encoder
label_encoder = LabelEncoder()

# Encode the 'Size' column
df["Size"] = label_encoder.fit_transform(df["Size"])

# Show the mapping
size_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Size Mapping:", size_mapping)

# Display the dataset again
from sklearn.model_selection import train_test_split

# Features (X) and target labels (y)
X = df[["Shoulder Distance", "Torso Height"]]
y = df["Size"]

from sklearn.model_selection import train_test_split

# Features (X) and target labels (y)
X = df[["Shoulder Distance", "Torso Height"]]
y = df["Size"]

# Split into 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")


from sklearn.neighbors import KNeighborsClassifier

# Initialize the k-NN classifier with k=3
model = KNeighborsClassifier(n_neighbors=3)
# Train your model
model.fit(X_train, y_train)

# Save the model
with open('D:/VTON/Model/knn_size_recommender.pkl', 'wb') as f:
    pickle.dump(model, f)

from sklearn.metrics import accuracy_score

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Example input (new person measurements)
new_measurement = [[115, 210]]  # [Shoulder Distance, Torso Height]

# Predict the size
predicted_size = model.predict(new_measurement)[0]

# Convert back to label
predicted_label = label_encoder.inverse_transform([predicted_size])[0]

print(f"Recommended Size: {predicted_label}")
import joblib

# Save the model
