import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import learning_curve

# Load the dataset
data = pd.read_csv("D:\MLProject\credit_rating.csv")

# Drop the S.No. columns
data = data.drop(data.columns[data.columns.str.contains('S.No')], axis=1)

# Handle missing values if any
data.dropna(inplace=True)

# Label encoding for categorical variables
label_encoder = LabelEncoder()
for col in data.columns[data.dtypes == 'object']:
    data[col] = label_encoder.fit_transform(data[col])

# Save feature names
feature_names = data.columns.tolist()

# Splitting the data into features (X) and target variable (y)
X = data.drop('Credit classification', axis=1)
y = data['Credit classification']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE for class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Logistic Regression with cross-validation
lr_model = LogisticRegression(penalty='l2', solver='liblinear')
cv_lr_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Scores for Logistic Regression:", cv_lr_scores)
print("Mean Accuracy (Logistic Regression):", np.mean(cv_lr_scores))

# Get LR Predictions
lr_model.fit(X_train, y_train)
lr_pred_prob = lr_model.predict_proba(X_test)

# Feature Engineering for Stacking
stacking_features = lr_pred_prob

# Train DecisionTreeClassifier on stacking features
dt_model = DecisionTreeClassifier()
dt_model.fit(stacking_features, y_test)

# Generate predictions from DecisionTreeClassifier
dt_pred_prob = dt_model.predict_proba(stacking_features)

# Augment stacking features with predictions from DecisionTreeClassifier
augmented_stacking_features = np.concatenate((stacking_features, dt_pred_prob), axis=1)

# Split data into training and validation sets for final model evaluation
X_train_final, X_val, y_train_final, y_val = train_test_split(augmented_stacking_features, y_test, test_size=0.2, random_state=42)

# Train final meta-model (Logistic Regression) using augmented features
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train_final, y_train_final)

# Print Best Parameters
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Use best model from GridSearchCV
final_lr_model = grid_search.best_estimator_

# Predict using the best model on validation set
final_pred_val = final_lr_model.predict(X_val)
print(final_pred_val)

# Calculate evaluation metrics for final stacked model on validation set
final_accuracy_val = accuracy_score(y_val, final_pred_val)
final_precision_val = precision_score(y_val, final_pred_val, average='weighted')
final_recall_val = recall_score(y_val, final_pred_val, average='weighted')
final_f1_val = f1_score(y_val, final_pred_val, average='weighted')

print("\nFinal Stacked Model Metrics on Validation Set:")
print("Accuracy:", final_accuracy_val)
print("Precision:", final_precision_val)
print("Recall:", final_recall_val)
print("F1 Score:", final_f1_val)

# Save the trained models
with open('lr_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

with open('dt_model.pkl', 'wb') as f:
    pickle.dump(dt_model, f)

with open('final_lr_model.pkl', 'wb') as f:
    pickle.dump(final_lr_model, f)

# Save the scaler object
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save the feature names
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
