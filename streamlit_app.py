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

# Styling
st.markdown("""
    <style>
        .stApp {
            background-color: #cce0ff;
        }
        .stTitle, h1, h2, h3 {
            color: #ff4500;
        }
        .stSelectbox label {
            color: #cc0000;
        }
        .stButton > button {
            background-color: #001a33;
            color: white;
            border-radius: 10px;
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

# Title and upload
st.markdown("""
    <div class="logo-container">
        <h1>🎨📊 ADS542 - Project Streamlit</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("### 🚀 Upload your dataset and explore the results!")

page = st.radio("Navigate to Model Explanations:", ['Model Performance', 'Logistic Regression', 'Random Forest', 'Neural Network'])

if page == 'Model Performance':
    uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, sep=';', quotechar='"')
        st.write("### 📜 Uploaded Data Preview")
        st.dataframe(df.head())

        target_column = st.selectbox("🎯 Select Target Column", df.columns)

        if target_column:
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

            # Feature selection
            model_rfe = LogisticRegression(max_iter=1000, random_state=42)
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
            ax.barh(top_features, importances[indices[:top_n]], color=['#001a33', '#ff4500', '#990000'])
            ax.set_xlabel("Feature Importance")
            ax.set_title("Top 10 Important Features")
            st.pyplot(fig)

            results = {}
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(),
                "Neural Network": MLPClassifier(max_iter=1000, early_stopping=True)
            }

            best_model, best_acc = None, 0

            for name, model in models.items():
                model.fit(X_train_preprocessed, y_train)
                y_pred = model.predict(X_test_preprocessed)

                if hasattr(model, "predict_proba"):
                    if len(np.unique(y_test)) > 2:
                        auc_value = roc_auc_score(y_test, model.predict_proba(X_test_preprocessed), multi_class="ovr", average="macro")
                    else:
                        auc_value = roc_auc_score(y_test, model.predict_proba(X_test_preprocessed)[:, 1])
                else:
                    auc_value = float('nan')

                acc = accuracy_score(y_test, y_pred)
                results[name] = {'Accuracy': acc, 'AUC': auc_value}
                if acc > best_acc:
                    best_acc = acc
                    best_model = model

            st.write("### 🏆 Model Performance")
            st.dataframe(pd.DataFrame(results).T)

            # Accuracy bar chart
            fig, ax = plt.subplots()
            ax.barh(list(results.keys()), [v['Accuracy'] for v in results.values()])
            ax.set_title("Model Accuracy Comparison")
            st.pyplot(fig)

            # ROC Curve
            st.markdown("### 📈 ROC Curve for Each Model 📉")
            fig, ax = plt.subplots()
            for name, model in models.items():
                try:
                    if len(np.unique(y_test)) > 2:
                        continue  # skip for multiclass ROC here for simplicity
                    y_prob = model.predict_proba(X_test_preprocessed)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
                except:
                    continue

            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend(loc='lower right')
            st.pyplot(fig)

            if best_model:
                with open("best_model.pkl", "wb") as f:
                    pickle.dump(best_model, f)
                st.success(f"🏅 Best Model: {max(results, key=lambda k: results[k]['Accuracy'])} with Accuracy: {best_acc:.2f}")

elif page == 'Logistic Regression':
    st.write("""
    ### 📘 Logistic Regression
    Logistic Regression estimates the probability of a binary response based on predictor variables.
    It's efficient for linearly separable data and provides interpretable coefficients.
    """)

elif page == 'Random Forest':
    st.write("""
    ### 📚 Random Forest
    Random Forest is an ensemble learning method that builds multiple decision trees and merges their output.
    It improves accuracy and reduces overfitting.
    """)

elif page == 'Neural Network':
    st.write("""
    ### 🤖 Neural Network
    Neural Networks are modeled after the human brain and are capable of learning complex patterns.
    They're useful in areas like image recognition and NLP.
    """)

