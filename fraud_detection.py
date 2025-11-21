import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Load CSV ---
print("Loading CSV file...")
df = pd.read_csv("creditcard.csv")
print(f"CSV loaded! Total rows: {len(df)}")

# --- Step 2: Prepare features and labels ---
print("Preparing features and labels...")
X = df.drop("Class", axis=1)
y = df["Class"]

# --- Step 3: Split data ---
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Step 4: Train Logistic Regression model ---
print("Training Logistic Regression model (this may take some time)...")
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
print("Model training completed!")

# --- Step 5: Test / Evaluate model ---
print("Making predictions on test set...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.6f}")
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# --- Step 6: Visualize data ---
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Class (0 = Non-Fraud, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.show()

# --- Step 7: Save trained model ---
joblib.dump(model, "fraud_model.pkl")
print("\nModel saved as 'fraud_model.pkl'")

print("\n✅ Fraud detection script finished successfully!")
