import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load cleaned dataset
df = pd.read_csv("cleaned_returns_data.csv")

# Convert categorical data
df = pd.get_dummies(df, drop_first=True)

# Split data
X = df.drop("Return_Status", axis=1)
y = df["Return_Status"]

# Save column names
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained successfully")
