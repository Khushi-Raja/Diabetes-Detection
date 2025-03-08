import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib

# 1. Load data
df = pd.read_csv("diabetes.csv")

# 2. Handle missing values (0s in medical data)
columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
imputer = SimpleImputer(missing_values=0, strategy='median')
df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])

# 3. Split data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Save model, scaler, and imputer
joblib.dump(model, 'model/diabetes_model.pkl')
joblib.dump(scaler, 'model/diabetes_scaler.pkl')
joblib.dump(imputer, 'model/diabetes_imputer.pkl')

# 7. Evaluate (optional)
print("Model, scaler, and imputer saved successfully!")