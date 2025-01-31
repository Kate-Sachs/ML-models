# ### Load Libraries
# Import necessary packages
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from xgboost import XGBClassifier
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from lightgbm import LGBMClassifier
import catboost as cb
from catboost import CatBoostClassifier
from tabulate import tabulate

# Get the current working directory
current_path = os.getcwd()
# Read CSV file into a DataFrame
BankChurners_df_all = pd.read_csv(current_path + '/datasets/'+'BankChurners.csv')
# Exclude the last 2 columns
BankChurners_df = BankChurners_df_all.iloc[:, :-2]

## Exploratory Analysis
#Preview the Data
BankChurners_df.head()

# Ckeck unique and duplicated vulues of dataframe
print(f"Check unique of CLIENTNUM = {len(BankChurners_df['CLIENTNUM'].unique())}")
print(f"Check duplicated of CLIENTNUM = {BankChurners_df['CLIENTNUM'].duplicated().sum()}")
print(f"Check duplicated of dataframe = {BankChurners_df.duplicated().sum()}")

# Drop the 'CLIENTNUM' column
BankChurners_df = BankChurners_df_all.iloc[:, :-2].drop(columns=['CLIENTNUM'])

# Checking the number of rows and columns
BankChurners_df.shape

# Calculate descriptive statistics
statistics = BankChurners_df.describe().T.style.bar(subset=['mean'], color='Purples')\
    .background_gradient(subset=['std'], cmap='viridis')\
    .background_gradient(subset=['75%'], cmap='viridis')\
    .background_gradient(subset=['max'], cmap='viridis')\
    .format(precision=2)  # Set the precision to 2 decimal places
# Display the output
statistics

# #### 1. Handling Outliers
# Checking for the number of null values present in each column
BankChurners_df.isnull().sum()

#### Counts 

flag_gender_counts = BankChurners_df.groupby(['Attrition_Flag', 'Gender']).size().reset_index(name='count')
print(flag_gender_counts)

flag_gender_counts_2 = BankChurners_df.groupby(['Attrition_Flag']).size().reset_index(name='count')
print(flag_gender_counts_2)


#################################
##### Class Imbalance Metric
#################################

# Calculate class proportions
class_proportions = BankChurners_df['Attrition_Flag'].value_counts(normalize=True)
# Calculate imbalance ratio
imbalance_ratio = class_proportions[0] / class_proportions[1]
print("Class Proportions:")
print(class_proportions) 
print("Imbalance Ratio (Class 0 / Class 1):", imbalance_ratio)

# ### One-Hot encoding
# Select categorical columns for one-hot encoding
cat_cols = ['Attrition_Flag','Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
# Create a OneHotEncoder object with `sparse_output=False`
encoder = OneHotEncoder(drop='first', sparse_output=False)
# Fit and transform the categorical columns
encoded_cols = encoder.fit_transform(BankChurners_df[cat_cols])
# Create column names for the encoded columns
encoded_columns = encoder.get_feature_names_out(cat_cols)
# Create a DataFrame of the encoded columns
encoded_df = pd.DataFrame(encoded_cols, columns=encoded_columns)
# Concatenate the encoded columns with the original DataFrame
BankChurners_df_encoded = pd.concat([BankChurners_df.drop(columns=cat_cols), encoded_df], axis=1)

#################################
##### Feature Selection
#################################

# Assuming X contains your feature matrix and y contains your target variable
X = BankChurners_df_encoded.drop('Attrition_Flag_Existing Customer', axis=1) # X features
y = BankChurners_df_encoded['Attrition_Flag_Existing Customer'] # Y target
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# **Decision Tree Classifier**

# Fit a decision tree classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Get feature importance scores
importance_scores = dt.feature_importances_
# Get the indices of the top 20 features
top20_indices = importance_scores.argsort()[-20:][::-1]
# Get the names of the top 10 features
top20_feature_names_dt = X.columns[top20_indices]
# Subset the feature matrix with the top 10 features
X_top20 = X[top20_feature_names_dt]
# Create a DataFrame with feature names and their importance scores
feature_importances_dt = pd.DataFrame({
    'Feature': top20_feature_names_dt,
    'Importance': importance_scores[top20_indices]
})
# Display the DataFrame
print(feature_importances_dt)

# Calculate cumulative importance
sorted_indices = importance_scores.argsort()[::-1]
sorted_importances = importance_scores[sorted_indices]
cumulative_importance = np.cumsum(sorted_importances)
# Find the number of features needed for cumulative importance of 95%
threshold = 0.95
n_important_features = np.where(cumulative_importance >= threshold * cumulative_importance[-1])[0][0] + 1
# Display the number of important features
print("Number of features needed to reach 95% importance:", n_important_features)
#sum(feature_importances_dt.Importance)
Top 11 features which reach 95% of importance.


# Fit a decision tree classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
# Get feature importance scores
importance_scores = dt.feature_importances_
# Get the indices of the top 11 features
top11_indices = importance_scores.argsort()[-n_important_features:][::-1]
# Get the names of the top 10 features
top11_feature_names_dt = X.columns[top11_indices]
# Subset the feature matrix with the top 10 features
X_top11 = X[top11_feature_names_dt]
# Create a DataFrame with feature names and their importance scores
feature_importances_dt = pd.DataFrame({
    'Feature': top11_feature_names_dt,
    'Importance': importance_scores[top11_indices]
})
# Plotting the importance scores of the top 11 features
plt.figure(figsize=(10, 6))
plt.barh(top11_feature_names_dt, importance_scores[top11_indices], color='magenta')
plt.xlabel('Feature Importance Score')
plt.ylabel('Feature Name')
plt.title('Top 11 Feature Importance Scores')
plt.gca().invert_yaxis()  # Invert y-axis to display the most important feature at the top
plt.show()


# **Random Forest  Classifier**

# Fit a random forest classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
# Get feature importance scores
importance_scores = rf.feature_importances_
# Get the indices of the top 20 features
top20_indices = importance_scores.argsort()[-20:][::-1]
# Get the names of the top 20 features
top20_feature_names_rf = X.columns[top20_indices]
# Subset the feature matrix with the top 20 features
X_top20 = X[top20_feature_names_rf]
# Create a DataFrame with feature names and their importance scores
feature_importances_rf = pd.DataFrame({
    'Feature': top20_feature_names_rf,
    'Importance': importance_scores[top20_indices]
})
# Display the DataFrame
print(feature_importances_rf)
# Calculate cumulative importance
sorted_indices = importance_scores.argsort()[::-1]
sorted_importances = importance_scores[sorted_indices]
cumulative_importance = np.cumsum(sorted_importances)
# Find the number of features needed for cumulative importance of 95%
threshold = 0.95
n_important_features = np.where(cumulative_importance >= threshold * cumulative_importance[-1])[0][0] + 1
# Display the number of important features
print("Number of features needed to reach 95% importance:", n_important_features)
# Fit a random forest classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
# Get feature importance scores
importance_scores = rf.feature_importances_
# Get the indices of the top 16 features
top16_indices = importance_scores.argsort()[-n_important_features:][::-1]
# Get the names of the top 16 features
top16_feature_names_rf = X.columns[top16_indices]
# Subset the feature matrix with the top 16 features
X_top16 = X[top16_feature_names_rf]
# Plotting the importance scores of the top 10 features
plt.figure(figsize=(10, 6))
plt.barh(top16_feature_names_rf, importance_scores[top16_indices], color='skyblue')
plt.xlabel('Feature Importance Score')
plt.ylabel('Feature Name')
plt.title('Top 16 Feature Importance Scores')
plt.gca().invert_yaxis()  # Invert y-axis to display the most important feature at the top
plt.show()

# **Gradient Boosting Machines (GBM)  Classifier**

# Fit a gradient boosting classifier
gbm = GradientBoostingClassifier(random_state=42)
gbm.fit(X_train, y_train)
# Get feature importance scores
importance_scores = gbm.feature_importances_
# Get the indices of the top 20 features
top20_indices = importance_scores.argsort()[-20:][::-1]
# Get the names of the top 10 features
top20_feature_names_gbm = X.columns[top20_indices]
# Subset the feature matrix with the top 20 features
X_top20 = X[top20_feature_names_gbm]
# Create a DataFrame with feature names and their importance scores
feature_importances_gbm = pd.DataFrame({
    'Feature': top20_feature_names_gbm,
    'Importance': importance_scores[top20_indices]
})
# Display the DataFrame
print(feature_importances_gbm)
# Calculate cumulative importance
sorted_indices = importance_scores.argsort()[::-1]
sorted_importances = importance_scores[sorted_indices]
cumulative_importance = np.cumsum(sorted_importances)
# Find the number of features needed for cumulative importance of 95%
threshold = 0.95
n_important_features = np.where(cumulative_importance >= threshold * cumulative_importance[-1])[0][0] + 1
# Display the number of important features
print("Number of features needed to reach 95% importance:", n_important_features)


# Fit a gradient boosting classifier
gbm = GradientBoostingClassifier(random_state=42)
gbm.fit(X_train, y_train)
# Get feature importance scores
importance_scores = gbm.feature_importances_
# Get the indices of the top 7 features
top7_indices = importance_scores.argsort()[-n_important_features:][::-1]
# Get the names of the top 7 features
top7_feature_names_gbm = X.columns[top7_indices]
# Subset the feature matrix with the top 7 features
X_top7 = X[top7_feature_names_gbm]
# Plotting the importance scores of the top 7 features
plt.figure(figsize=(10, 6))
plt.barh(top7_feature_names_gbm, importance_scores[top7_indices], color='#782170')
plt.xlabel('Feature Importance Score')
plt.ylabel('Feature Name')
plt.title('Top 7 Feature Importance Scores')
plt.gca().invert_yaxis()  # Invert y-axis to display the most important feature at the top
plt.show()

#################################
##### K-fold Cross Validation
#################################

# Setup K-Folds
kf = KFold(n_splits=10, shuffle=True, random_state=42)
# List of classifiers
classifiers = {
    "Decision Tree": dt,
    "Random Forest": rf,
    "GBM": gbm
}
# Metrics to evaluate
metrics = ['accuracy', 'precision', 'recall', 'f1']
# Dictionary to hold scores
results = {metric: {name: None for name in classifiers} for metric in metrics}
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
for metric in metrics:
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X, y, cv=kf, scoring=metric)
        results[metric][name] = f"{scores.mean():0.2f} (+/- {scores.std() * 2:0.2f})"

df_results = pd.DataFrame(results)
df_results.index.name = 'Metric'
print(df_results)
# Fit a gradient boosting classifier
gbm = GradientBoostingClassifier(random_state=42)
gbm.fit(X_train, y_train)
# Assuming X_test and y_test are defined
y_pred = gbm.predict(X_test)
# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gbm.classes_)
disp.plot()
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))



#################################
##### ML Models
#################################

##### Standard

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Reset the indices of the train and test sets
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to NumPy array if necessary
X_train_scaled = np.array(X_train_scaled)
y_train = np.array(y_train)

# List of models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "Bayesian Network": BernoulliNB(),
    "k-Nearest Neighbor": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(verbose=-1),
    "CatBoost": CatBoostClassifier(verbose=0)
}

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Function to evaluate a model and return metrics
def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_accuracy = []
    train_precision = []
    train_recall = []
    train_f1 = []
    train_roc_auc = []
    
    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        y_pred_proba = model.predict_proba(X_val_fold)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        train_accuracy.append(accuracy_score(y_val_fold, y_pred))
        train_precision.append(precision_score(y_val_fold, y_pred))
        train_recall.append(recall_score(y_val_fold, y_pred))
        train_f1.append(f1_score(y_val_fold, y_pred))
        train_roc_auc.append(roc_auc_score(y_val_fold, y_pred_proba))
    
    # Train on the full training set and evaluate on the test set
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_test_pred
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    metrics = {
        "Train Accuracy": np.mean(train_accuracy),
        "Train Precision": np.mean(train_precision),
        "Train Recall": np.mean(train_recall),
        "Train F1 Score": np.mean(train_f1),
        "Train ROC AUC": np.mean(train_roc_auc),
        "Test Accuracy": test_accuracy,
        "Test Precision": test_precision,
        "Test Recall": test_recall,
        "Test F1 Score": test_f1,
        "Test ROC AUC": test_roc_auc
    }
    
    return metrics

# Lists to store the results
train_results = []
test_results = []

# Evaluate each model
for model_name, model in models.items():
    print(f"Evaluating {model_name}...")
    metrics = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
    
    train_results.append([model_name, metrics["Train Accuracy"], metrics["Train Precision"], metrics["Train Recall"], metrics["Train F1 Score"], metrics["Train ROC AUC"]])
    test_results.append([model_name, metrics["Test Accuracy"], metrics["Test Precision"], metrics["Test Recall"], metrics["Test F1 Score"], metrics["Test ROC AUC"]])

# Create DataFrames from the results
train_results_df = pd.DataFrame(train_results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']).round(2)
test_results_df = pd.DataFrame(test_results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']).round(2)

# Display the DataFrames
print("\nTrain Results:")
#print(tabulate(train_results_df, headers='keys', tablefmt='psql'))
print(train_results_df)
print("\nTest Results:")
print(test_results_df)


##### IMBALANCE - SMOTE

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
# Convert to NumPy array if necessary
X_train_smote = np.array(X_train_smote)
y_train_smote = np.array(y_train_smote)

# List of models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "Bayesian Network": BernoulliNB(),
    "k-Nearest Neighbor": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(verbose=-1),
    "CatBoost": CatBoostClassifier(verbose=0)
}
# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# Function to evaluate a model and return metrics
def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_accuracy = []
    train_precision = []
    train_recall = []
    train_f1 = []
    train_roc_auc = []
    
    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        y_pred_proba = model.predict_proba(X_val_fold)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        train_accuracy.append(accuracy_score(y_val_fold, y_pred))
        train_precision.append(precision_score(y_val_fold, y_pred))
        train_recall.append(recall_score(y_val_fold, y_pred))
        train_f1.append(f1_score(y_val_fold, y_pred))
        train_roc_auc.append(roc_auc_score(y_val_fold, y_pred_proba))
    
    # Train on the full training set and evaluate on the test set
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_test_pred
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    metrics = {
        "Train Accuracy": np.mean(train_accuracy),
        "Train Precision": np.mean(train_precision),
        "Train Recall": np.mean(train_recall),
        "Train F1 Score": np.mean(train_f1),
        "Train ROC AUC": np.mean(train_roc_auc),
        "Test Accuracy": test_accuracy,
        "Test Precision": test_precision,
        "Test Recall": test_recall,
        "Test F1 Score": test_f1,
        "Test ROC AUC": test_roc_auc
    }
    
    return metrics
# Lists to store the results
train_results = []
test_results = []

# Evaluate each model
for model_name, model in models.items():
    print(f"Evaluating {model_name}...")
    metrics = evaluate_model(model, X_train_smote, y_train_smote, X_test_scaled, y_test)
    
    train_results.append([model_name, metrics["Train Accuracy"], metrics["Train Precision"], metrics["Train Recall"], metrics["Train F1 Score"], metrics["Train ROC AUC"]])
    test_results.append([model_name, metrics["Test Accuracy"], metrics["Test Precision"], metrics["Test Recall"], metrics["Test F1 Score"], metrics["Test ROC AUC"]])

# Create DataFrames from the results
train_results_df = pd.DataFrame(train_results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']).round(2)
test_results_df = pd.DataFrame(test_results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']).round(2)
# Display the DataFrames
print("\nTrain Results:")
print(train_results_df)
print("\nTest Results:")
print(test_results_df)


##### FEATURE SELECTION
# provided dataset and preprocessing code
top7_features = ['Total_Trans_Ct', 'Total_Trans_Amt', 'Total_Revolving_Bal', 'Total_Ct_Chng_Q4_Q1', 'Total_Relationship_Count', 'Total_Amt_Chng_Q4_Q1', 'Customer_Age']
X_top7 = BankChurners_df_encoded[top7_features]
y = BankChurners_df_encoded['Attrition_Flag_Existing Customer'] 

# Split the dataset using only the top 7 features
X_train, X_test, y_train, y_test = train_test_split(X_top7, y, test_size=0.3, random_state=42, stratify=y)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# List of models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "Bayesian Network": BernoulliNB(),
    "k-Nearest Neighbor": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(verbose=-1),
    "CatBoost": CatBoostClassifier(verbose=0)
}

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Function to evaluate a model and return metrics
def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_accuracy = []
    train_precision = []
    train_recall = []
    train_f1 = []
    train_roc_auc = []
    
    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        y_pred_proba = model.predict_proba(X_val_fold)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        train_accuracy.append(accuracy_score(y_val_fold, y_pred))
        train_precision.append(precision_score(y_val_fold, y_pred))
        train_recall.append(recall_score(y_val_fold, y_pred))
        train_f1.append(f1_score(y_val_fold, y_pred))
        train_roc_auc.append(roc_auc_score(y_val_fold, y_pred_proba))
    
    # Train on the full training set and evaluate on the test set
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_test_pred
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    metrics = {
        "Train Accuracy": np.mean(train_accuracy),
        "Train Precision": np.mean(train_precision),
        "Train Recall": np.mean(train_recall),
        "Train F1 Score": np.mean(train_f1),
        "Train ROC AUC": np.mean(train_roc_auc),
        "Test Accuracy": test_accuracy,
        "Test Precision": test_precision,
        "Test Recall": test_recall,
        "Test F1 Score": test_f1,
        "Test ROC AUC": test_roc_auc
    }
    
    return metrics

# Lists to store the results
train_results = []
test_results = []
# Evaluate each model
for model_name, model in models.items():
    print(f"Evaluating {model_name}...")
    metrics = evaluate_model(model, X_train_smote, y_train_smote, X_test_scaled, y_test)
    
    train_results.append([model_name, metrics["Train Accuracy"], metrics["Train Precision"], metrics["Train Recall"], metrics["Train F1 Score"], metrics["Train ROC AUC"]])
    test_results.append([model_name, metrics["Test Accuracy"], metrics["Test Precision"], metrics["Test Recall"], metrics["Test F1 Score"], metrics["Test ROC AUC"]])
# Create DataFrames from the results
train_results_df = pd.DataFrame(train_results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']).round(3)
test_results_df = pd.DataFrame(test_results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']).round(3)
# Display the DataFrames
print("\nTrain Results:")
print(train_results_df)
print("\nTest Results:")
print(test_results_df)
# Generate confusion matrix for CatBoost model
catboost_model = CatBoostClassifier(verbose=0)
catboost_model.fit(X_train_smote, y_train_smote)
y_pred = catboost_model.predict(X_test_scaled)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=catboost_model.classes_)
disp.plot()
plt.title('Confusion Matrix - CatBoost')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()