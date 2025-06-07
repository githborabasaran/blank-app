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
import os
import joblib
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve, auc
import imblearn
import sklearn
from imblearn.under_sampling import RandomUnderSampler

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

st.markdown(f"""
    <div class="logo-container">
       
        <h1>🎨📊 ADS542 - Project Streamlit</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("### 🚀 Upload your dataset and explore the results!")

# Add navigation buttons for model explanation sections
page = st.radio("Navigate to Model Explanations:", ['Model Performance', 'Logistic Regression', 'Random Forest', 'Neural Network'])
# The page condition
# The page condition
if page == 'Model Performance':
    # ✅ Use hardcoded file path
    file_path = 'bank-additional.csv'

    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep=';', quotechar='"')
        st.write("### 📜 Uploaded Data Preview")
        st.dataframe(df.head())

        target_column = 'y'
        if target_column not in df.columns:
            st.error(f"🚫 The target column '{target_column}' does not exist in the dataset!")
        else:
            # Preprocess data
            df, preprocessor, num_features, cat_features = preprocess_data(df, target_column)

            X = df.drop(columns=[target_column])
            y = df[target_column]

            if y.dtypes == 'object':
                y = LabelEncoder().fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            pipeline = Pipeline([('preprocessor', preprocessor)])

            X_train_preprocessed = pipeline.fit_transform(X_train)
            X_test_preprocessed = pipeline.transform(X_test)

            under_sampler = RandomUnderSampler(random_state=42)
            X_train_res, y_train_res = under_sampler.fit_resample(np.array(X_train_preprocessed), np.array(y_train))
            # Feature names extraction
            feature_names = []
            for name, transformer, columns in preprocessor.transformers_:
                if hasattr(transformer, 'get_feature_names_out'):
                    feature_names.extend(transformer.get_feature_names_out(columns))
                else:
                    feature_names.extend(columns)

            # Feature selection with RFE
            model_rfe = LogisticRegression(random_state=42)
            selector = RFE(model_rfe, n_features_to_select=min(10, X_train_preprocessed.shape[1]))
            X_train_rfe = selector.fit_transform(X_train_preprocessed, y_train_res)

            # Random Forest for feature importance
            rf_model = RandomForestClassifier(random_state=42)
            rf_model.fit(X_train_preprocessed, y_train_res)

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

            # Train and evaluate models
            results = {}
            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "Neural Network": MLPClassifier(max_iter=1000, solver='adam', early_stopping=True, random_state=42)
            }

            best_model, best_acc = None, 0

            for name, model in models.items():
                model.fit(X_train_preprocessed, y_train_res)
                y_pred = model.predict(X_test_preprocessed)
                y_proba = model.predict_proba(X_test_preprocessed)

                acc = accuracy_score(y_test, y_pred)

                if len(np.unique(y_test)) == 2:
                    auc_value = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    num_classes = len(np.unique(y_test))
                    if y_proba.shape[1] != num_classes:
                        st.warning(f"{name} returned {y_proba.shape[1]} prob columns, expected {num_classes}")
                        auc_value = np.nan
                    else:
                        auc_value = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

                results[name] = {'Accuracy': acc, 'AUC': auc_value}

                if acc > best_acc:
                    best_acc = acc
                    best_model = model

            # Accuracy plot section
            st.markdown("### 📊 Model Accuracy Comparison 🏅")
            st.write("The bar chart below shows the accuracy of each model evaluated.")

            fig, ax = plt.subplots(figsize=(10, 6))
            model_names = list(results.keys())
            accuracies = [results[name]['Accuracy'] for name in model_names]

            ax.barh(model_names, accuracies, color=['#001a33', '#ff4500', '#990000', '#ffa500', '#33cc33'])
            ax.set_xlabel('Accuracy', color='#001a33')
            ax.set_title('Model Accuracy Comparison', color='#cc0000')
            st.pyplot(fig)

            # ROC curve plot section
            st.markdown("### 📈 ROC Curve for Each Model 📉")
            st.write("The ROC curve below compares the true positive rate (TPR) and false positive rate (FPR) of each model.")

            fig, ax = plt.subplots(figsize=(10, 6))
            for name, model in models.items():
                y_score = model.predict_proba(X_test_preprocessed)

                if len(np.unique(y_test)) == 2:
                    fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
                else:
                    from sklearn.preprocessing import label_binarize
                    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
                    if y_score.shape[1] != y_test_bin.shape[1]:
                        continue
                    for i in range(y_test_bin.shape[1]):
                        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, label=f'{name} - Class {i} (AUC = {roc_auc:.2f})')

            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve Comparison')
            ax.legend(loc='lower right')
            st.pyplot(fig)

            # Save the best model to disk
            if best_model:
                with open("best_model.pkl", "wb") as f:
                    pickle.dump(best_model, f)
                st.success(f"🏅 Best Model: {max(results, key=lambda k: results[k]['Accuracy'])} with Accuracy: {best_acc:.2f}")

            # --- Prediction form section ---
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
                'previous': [0],  # Default value
                'emp.var.rate': [1.1],
                'poutcome': ['nonexistent'],
                'euribor3m': [4.5],
                'month': ['may'],
                'cons.price.idx': [93.2],
                'cons.conf.idx': [-40.0],
                'marital': ['married']
            })

            processed_input = preprocessor.transform(input_df)

            # Load best model for prediction
            model = None
            try:
                with open("best_model.pkl", "rb") as f:
                    model = pickle.load(f)
            except Exception as e:
                st.error("Failed to load the best model for prediction.")
                st.error(str(e))

            if model:
                if st.button("Predict Credit Approval"):
                    prediction = model.predict(processed_input)
                    if prediction[0] == 1:
                        st.success("✅ Credit Approved!")
                    else:
                        st.error("❌ Credit Not Approved.")

                    probability = model.predict_proba(processed_input)[0][1]
                    credit_score = int(380 + (probability * 550))  # Scale score from 300-850

                    st.write(f"🧮 Estimated Credit Score: **{credit_score}**")

                    if credit_score >= 500:
                        st.success("💚 Excellent credit score!")
                    elif credit_score >= 450:
                        st.info("💛 Good credit score.")
                    elif credit_score >= 400:
                        st.warning("🧡 Fair credit score.")
                    else:
                        st.error("❤️ Poor credit score.")

    else:
        st.error(f"🚫 File not found at: {file_path}")
        results={}


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




