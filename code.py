from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df = pd.read_csv("/Users/jheeljhala/Desktop/Projects/Lung Cancer/dataset_med.csv")


print(df.head()) #This is to view the different features in the dataset

#THE MAIN CODE or MODEL

df_clean = df.drop(['id', 'diagnosis_date', 'end_treatment_date'], axis=1)
df_encoded = pd.get_dummies(df_clean, drop_first=True)

X = df_encoded.drop('survived', axis=1).values
y = df_encoded['survived'].values

# Step 1: First split into temp (80%) and test (20%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

# Step 2: Split temp into training (60%) and validation (20%)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=23)


lung_cancer_clf = LogisticRegression(max_iter=1000, random_state=0)
lung_cancer_clf.fit(X_train, y_train)

acc = accuracy_score(y_test, lung_cancer_clf.predict(X_test)) * 100
print(f"Logistic Regression model accuracy: {acc:.2f}%")

#An example where I have given the model data on a random patient and the model outputs whether the patient will survive or not (1 or 0)

new_patient = {
    'age': [55],
    'gender': ['female'],
    'country': ['India'],
    'diagnosis_date': ['2023-04-20'],           
    'cancer_stage': [1],
    'family_history': [0],
    'smoking_status': ['Former Smoker'],
    'bmi': [30],
    'cholesterol_level': [190],
    'hypertension': [1],
    'asthma': [0],
    'cirrhosis': [0],
    'other_cancer': [0],
    'treatment_type': ['Chemotherapy'],
    'end_treatment_date': ['2025-03-22']}

new_df = pd.DataFrame(new_patient)

new_df_encoded = pd.get_dummies(new_df)
 

new_df_encoded = df_encoded.drop('survived', axis=1).values

prediction = lung_cancer_clf.predict(new_df_encoded)
print("Prediction (0 = did not survive, 1 = survived):", prediction[0])
