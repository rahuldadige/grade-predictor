# model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Load your updated data
df = pd.read_csv("student_performance.csv")

# 2. Preprocess
le_gender = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])

le_support = LabelEncoder()
df["ParentalSupport"] = le_support.fit_transform(df["ParentalSupport"])

# 3. Split features / target
X = df.drop(["StudentID", "Name", "FinalGrade"], axis=1)
y = df["FinalGrade"]

# 4. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE:", rmse)

# 7. Save model + encoders
joblib.dump(model, "grade_model.pkl")
joblib.dump(le_gender, "gender_encoder.pkl")
joblib.dump(le_support, "support_encoder.pkl")

print("✅ Model and encoders saved to disk.")
