import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler
import os

# --- Streamlit theme ---
st.markdown("""
    <style>
        body {
            background-color: #cce0ff;
            color: #001a33;
        }
        .stApp {
            background-color: #cce0ff;
        }
        h1 {
            color: #ff4500;
            font-weight: bold;
        }
        .stSelectbox label, .stSlider label {
            color: #cc0000;
            font-size: 18px;
        }
        .stButton > button {
            background-color: #001a33;
            color: white;
            border-radius: 10px;
            font-size: 16px;
            padding: 10px;
        }
        .stDataFrame {
            border: 3px solid #cc0000;
            border-radius: 10px;
            background-color: #f9f9f9;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🎨📊 ADS542 - Project Streamlit")

st.markdown("### 🚀 Upload your dataset and explore the results!")

# --- Load dataset ---
file_path = 'bank-additional.csv'

if not os.path.exists(file_path):
    st.error(f"File '{file_path}' not found! Please upload your dataset or check the file path.")
    st.stop()

df = pd.read_csv(file_path, sep=';', quotechar='"')
st.write("### 📜 Dataset Preview")
st.dataframe(df.head())

target_column = 'y'
if target_column not in df.columns:
    st.error(f"🚫 The target column '{target_column}' does not exist in the dataset!")
    st.stop()

# --- Preprocessing ---

# Clean data
df = df.dropna().drop_duplicates()

# Identify numeric and categorical features
num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Remove target column from features list
if target_column in num_features:
    num_features.remove(target_column)
if target_column in cat_features:
    cat_features.remove(target_column)

# Build preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
])

# Split features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Encode target if needed
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit preprocessor only on training set
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# --- Undersample ---
under_sampler = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = under_sampler.fit_resample(X_train_preprocessed, y_train)

# --- Extract feature names ---
feature_names = []

# numeric feature names
feature_names.extend(num_features)

# categorical feature names from onehot encoder
ohe = preprocessor.named_transformers_['cat']
cat_feature_names = ohe.get_feature_names_out(cat_features)
feature_names.extend(cat_feature_names.tolist())

# --- Feature Selection with RFE ---
rfe_model = LogisticRegression(random_state=42, max_iter=1000)
selector = RFE(rfe_model, n_features_to_select=min(10, X_train_res.shape[1]))
X_train_rfe = selector.fit_transform(X_train_res, y_train_res)

# --- Random Forest for feature importance ---
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_res, y_train_res)
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

top_n = min(10, len(feature_names))
top_features = [feature_names[i] for i in indices[:top_n]]

st.write("### 🌟 Top 10 Important Features")
st.dataframe(pd.DataFrame({'Feature': top_features, 'Importance': importances[indices[:top_n]]}))

fig, ax = plt.subplots()
ax.barh(top_features[::-1], importances[indices[:top_n]][::-1], color=['#001a33', '#ff4500', '#990000', '#ffa500', '#33cc33'])
ax.set_xlabel("Feature Importance", color="#001a33")
ax.set_title("Top 10 Important Features", color="#cc0000")
st.pyplot(fig)

# --- Train models ---
results = {}
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Neural Network": MLPClassifier(max_iter=1000, solver='adam', early_stopping=True, random_state=42)
}

best_model = None
best_acc = 0

for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_preprocessed)
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_preprocessed)
    else:
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    auc_value = np.nan
    if y_proba is not None:
        if len(np.unique(y_test)) == 2:
            auc_value = roc_auc_score(y_test, y_proba[:, 1])
        else:
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
            if y_proba.shape[1] == y_test_bin.shape[1]:
                auc_value = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='macro')

    results[name] = {'Accuracy': acc, 'AUC': auc_value}

    if acc > best_acc:
        best_acc = acc
        best_model = model

# --- Display model accuracy ---
st.markdown("### 📊 Model Accuracy Comparison 🏅")
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(list(results.keys()), [r['Accuracy'] for r in results.values()], color=['#001a33', '#ff4500', '#990000'])
ax.set_xlabel('Accuracy', color='#001a33')
ax.set_title('Model Accuracy Comparison', color='#cc0000')
st.pyplot(fig)

# --- ROC curves ---
st.markdown("### 📈 ROC Curve for Each Model 📉")
fig, ax = plt.subplots(figsize=(8, 6))

for name, model in models.items():
    if not hasattr(model, "predict_proba"):
        continue

    y_proba = model.predict_proba(X_test_preprocessed)

    if len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    else:
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        if y_proba.shape[1] != y_test_bin.shape[1]:
            continue
        for i in range(y_test_bin.shape[1]):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} Class {i} (AUC = {roc_auc:.2f})")

ax.plot([0,1], [0,1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves')
ax.legend()
st.pyplot(fig)

# Add a section for the ROC curve plot
st.markdown("### 📈 ROC Curve for Each Model 📉")
st.write("The ROC curve below compares the true positive rate (TPR) and false positive rate (FPR) of each model.")

# Plot AUC Curve for Each Model
fig, ax = plt.subplots(figsize=(10, 6))

for name, model in models.items():
    # Compute the ROC curve for the current model
    try:
        if len(np.unique(y_test)) > 2:  # Multi-class classification
            # One-vs-Rest (OvR) approach for multi-class ROC curve
            fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_preprocessed), pos_label=None)
            auc_value = auc(fpr, tpr)
        else:  # Binary classification
            fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_preprocessed)[:, 1])
            auc_value = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc_value:.2f})')

    except Exception as e:
        st.warning(f"Error computing ROC curve for {name}: {e}")

ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
ax.set_xlabel('False Positive Rate', color='#001a33')
ax.set_ylabel('True Positive Rate', color='#001a33')
ax.set_title('Receiver Operating Characteristic (ROC) Curve', color='#cc0000')
ax.legend(loc='lower right')
st.pyplot(fig)

# Display the best model after the plots
st.success(f"🏅 Best Model: {max(results, key=lambda k: results[k]['Accuracy'])} with Accuracy: {best_acc:.2f}")



model = joblib.load("best_model.pkl")  # Ensure you saved it previously

st.subheader("Enter your information below:")

age = st.slider("Age", 18, 100, 30)
job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
                           'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
education = st.selectbox("Education", ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 
                                       'illiterate', 'professional.course', 'university.degree', 'unknown'])
default = st.selectbox("Has Credit in Default?", ['yes', 'no'])
housing = st.selectbox("Has Housing Loan?", ['yes', 'no'])
loan = st.selectbox("Has Personal Loan?", ['yes', 'no'])
contact = st.selectbox("Contact Communication Type", ['cellular', 'telephone'])
dayofweek = st.selectbox("Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
duration = st.number_input("Last Contact Duration (seconds)", min_value=0, value=100)
campaign = st.number_input("Number of Contacts During Campaign", min_value=1, value=1)
pdays = st.number_input("Days Since Last Contact", min_value=-1, value=-1)
nr_employed = st.number_input("Number of Employees (Economic Indicator)", min_value=0.0, value=5000.0)

# Add missing columns with defaults (assume safest default or most common values)
input_df = pd.DataFrame({
    'age': [age],
    'job': [job],
    'education': [education],
    'default': [default],
    'housing': [housing],
    'loan': [loan],
    'contact': [contact],
    'day_of_week': [dayofweek],
    'duration': [duration],
    'campaign': [campaign],
    'pdays': [pdays],
    'nr.employed': [nr_employed],
    'previous': [0],  # Default to 0 previous contacts
    'emp.var.rate': [1.1],  # Example average value; adjust if known
    'poutcome': ['nonexistent'],  # Most common value for this field
    'euribor3m': [4.5],  # Approximate average; adjust as needed
    'month': ['may'],  # Default to most frequent month
    'cons.price.idx': [93.2],  # Example value
    'cons.conf.idx': [-40.0],  # Example value
    'marital': ['married']  # Common marital status
})

# Optional: If you saved expected columns during training, you can reindex:
# input_df = input_df.reindex(columns=expected_columns)

# Apply preprocessing
processed_input = preprocessor.transform(input_df)

# Predict
if st.button("Predict Credit Approval"):
    prediction = model.predict(processed_input)
    if prediction[0] == 1:
        st.success("✅ Credit Approved!")
    else:
        st.error("❌ Credit Not Approved.")

# Get prediction probability (for class 1 - approval)
probability = model.predict_proba(processed_input)[0][1]

# Scale probability to a credit score range (e.g., 300 to 850)
credit_score = int(380 + (probability * 550))  # 550 = 850 - 300

st.write(f"🧮 Estimated Credit Score: **{credit_score}**")

# Optional: Add rating explanation
if credit_score >= 500:
    st.success("💚 Excellent credit score!")
elif credit_score >= 450:
    st.info("💛 Good credit score.")
elif credit_score >= 400:
    st.warning("🧡 Fair credit score.")
else:
    st.error("❤️ Poor credit score.")
