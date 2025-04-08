import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve, auc

# Set Streamlit theme with vibrant colors
st.markdown("""
    <style>
        body {
            background-color: #cce0ff;
            color: #001a33;
        }
        .stApp {
            background-color: #cce0ff;
        }
        .stTitle {
            color: #ff4500;
            font-size: 36px;
            font-weight: bold;
        }
        .stSelectbox label {
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

def preprocess_data(df, target_column):
    df = df.dropna().drop_duplicates()
    num_features = df.select_dtypes(include=['int64', 'float64']).columns.to_list()
    cat_features = df.select_dtypes(include=['object', 'category']).columns.to_list()
    
    if target_column in num_features:
        num_features.remove(target_column)
    if target_column in cat_features:
        cat_features.remove(target_column)
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ])
    
    return df, preprocessor, num_features, cat_features

# Navigation Buttons
st.markdown("""
    <style>
        .logo-container {
            display: flex;
            align-items: center;
        }
        .logo-container img {
            width: 70px;
            margin-right: 60px;
        }
        .logo-container h1 {
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)

# Display title with logo
logo_path = '/Users/nilbasaran/Desktop/tedu_image/tedu.gif'  # Replace with your logo file path
st.markdown(f"""
    <div class="logo-container">
        <img src="{logo_path}" alt="Logo">
        <h1>🎨📊 ADS542 - Project Streamlit</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("### 🚀 Upload your dataset and explore the results!")

# Add navigation buttons for model explanation sections
page = st.radio("Navigate to Model Explanations:", ['Model Performance', 'Logistic Regression', 'Random Forest', 'Neural Network'])

if page == 'Model Performance':
    uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file, sep=';', quotechar='"')
        st.write("### 📜 Uploaded Data Preview")
        st.dataframe(df.head())

        target_column = st.selectbox("🎯 Select Target Column", df.columns)
        df, preprocessor, num_features, cat_features = preprocess_data(df, target_column)

        X = df.drop(columns=[target_column])
        y = df[target_column]

        if y.dtypes == 'object':
            y = LabelEncoder().fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipeline = Pipeline([
            ('preprocessor', preprocessor)
        ])

        X_train_preprocessed = pipeline.fit_transform(X_train)
        X_test_preprocessed = pipeline.transform(X_test)

        feature_names = []
        for name, transformer, columns in preprocessor.transformers_:
            if hasattr(transformer, 'get_feature_names_out'):
                feature_names.extend(transformer.get_feature_names_out(columns))
            else:
                feature_names.extend(columns)

        model_rfe = LogisticRegression(random_state=42)
        selector = RFE(model_rfe, n_features_to_select=min(10, X_train_preprocessed.shape[1]))
        X_train_rfe = selector.fit_transform(X_train_preprocessed, y_train)

        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train_preprocessed, y_train)

        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]

        top_n = min(10, len(feature_names))
        top_features = [feature_names[i] for i in indices[:top_n]]

        st.write("### 🌟 Top 10 Important Features")
        st.dataframe(pd.DataFrame({'Feature': top_features, 'Importance': importances[indices[:top_n]]}))

        fig, ax = plt.subplots()
        ax.barh(top_features, importances[indices[:top_n]], color=['#001a33', '#ff4500', '#990000', '#ffa500', '#33cc33'])
        ax.set_xlabel("Feature Importance", color="#001a33")
        ax.set_title("Top 10 Important Features", color="#cc0000")
        st.pyplot(fig)

        results = {}
        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(),
            "Neural Network": MLPClassifier(max_iter=1000, solver='adam', early_stopping=True, random_state=42)
        }

        best_model, best_acc = None, 0

        for name, model in models.items():
            model.fit(X_train_preprocessed, y_train)
            if len(np.unique(y_test)) > 2:
                auc_value = roc_auc_score(y_test, model.predict_proba(X_test_preprocessed), multi_class="ovr", average="macro")
            else:
                auc_value = roc_auc_score(y_test, model.predict_proba(X_test_preprocessed)[:, 1])
            acc = accuracy_score(y_test, model.predict(X_test_preprocessed))
            results[name] = {'Accuracy': acc, 'AUC': auc_value}

            if acc > best_acc:
                best_acc = acc
                best_model = model

        st.write("### 🏆 Model Performance")
        st.dataframe(pd.DataFrame(results).T)

        if best_model:
            with open("best_model.pkl", "wb") as f:
                pickle.dump(best_model, f)
            st.success(f"🏅 Best Model: {max(results, key=lambda k: results[k]['Accuracy'])} with Accuracy: {best_acc:.2f}")

elif page == 'Logistic Regression':
    st.write("""
    ### 📘 Logistic Regression
    Logistic Regression is a statistical method used for binary classification. 
    It estimates the probability of a binary response based on one or more predictor variables.
    
    **Key points:**
    - It uses the logistic function to model the probability of the default class (usually 1).
    - It is widely used due to its simplicity and efficiency for binary classification tasks.
    - The model provides a probabilistic output which can be converted to class labels.
    - Logistic Regression works best with linearly separable data.
    """)

elif page == 'Random Forest':
    st.write("""
    ### 📚 Random Forest
    Random Forest is an ensemble learning method that constructs multiple decision trees and merges them together 
    to improve accuracy and control overfitting.
    
    **Key points:**
    - It reduces overfitting by averaging the predictions of multiple trees.
    - It can be used for both classification and regression tasks.
    - Random Forest is a powerful algorithm that works well on large datasets with a high-dimensional feature space.
    - The final prediction is made by taking a majority vote (for classification) or averaging the outputs (for regression).
    """)

elif page == 'Neural Network':
    st.write("""
    ### 🤖 Neural Network
    A Neural Network is a model inspired by the human brain that learns patterns from data through a series of interconnected nodes (neurons).
    
    **Key points:**
    - It is composed of layers: input layer, hidden layers, and output layer.
    - Neural Networks are highly flexible and can model complex non-linear relationships in data.
    - They are especially useful in tasks like image recognition, speech processing, and natural language processing.
    - Training a neural network involves adjusting the weights between the neurons to minimize the error.
    """)
# Add a section for the accuracy plot
st.markdown("### 📊 Model Accuracy Comparison 🏅")
st.write("The bar chart below shows the accuracy of each model evaluated.")

# Plot Accuracy of Each Model
fig, ax = plt.subplots(figsize=(10, 6))
model_names = list(results.keys())
accuracies = [results[name]['Accuracy'] for name in model_names]

ax.barh(model_names, accuracies, color=['#001a33', '#ff4500', '#990000', '#ffa500', '#33cc33'])
ax.set_xlabel('Accuracy', color='#001a33')
ax.set_title('Model Accuracy Comparison', color='#cc0000')
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

