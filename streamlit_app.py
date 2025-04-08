import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import numpy as np

st.set_page_config(layout="wide")
st.title("🧠 AutoML - Model Evaluation")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset", df.head())

    target_column = st.selectbox("Select Target Column", df.columns)

    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encoding categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        n_features = st.slider("Select Number of Top Features (RFE)", min_value=1, max_value=min(20, X.shape[1]), value=10)
        rfe_selector = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=n_features)
        rfe_selector.fit(X_train_scaled, y_train)
        selected_features = X.columns[rfe_selector.support_]

        st.write("### 🔍 Top Important Features")
        st.write(selected_features.tolist())

        X_train_top = X_train[selected_features]
        X_test_top = X_test[selected_features]

        scaler = StandardScaler()
        X_train_preprocessed = scaler.fit_transform(X_train_top)
        X_test_preprocessed = scaler.transform(X_test_top)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Neural Network": MLPClassifier(max_iter=1000)
        }

        results = {}
        best_model = None
        best_acc = 0

        with st.spinner("⏳ Training Models... Please wait..."):
            for name, model in models.items():
                st.write(f"Training {name}...")
                try:
                    model.fit(X_train_preprocessed, y_train)
                    if len(np.unique(y_test)) > 2:
                        auc_value = roc_auc_score(
                            y_test, model.predict_proba(X_test_preprocessed), multi_class="ovr", average="macro")
                    else:
                        auc_value = roc_auc_score(
                            y_test, model.predict_proba(X_test_preprocessed)[:, 1])
                    acc = accuracy_score(y_test, model.predict(X_test_preprocessed))
                    results[name] = {'Accuracy': acc, 'AUC': auc_value}
                    if acc > best_acc:
                        best_acc = acc
                        best_model = model
                except Exception as e:
                    st.warning(f"⚠️ Error training {name}: {e}")

        if results:
            st.write("### 🏆 Model Performance")
            st.dataframe(pd.DataFrame(results).T)

            if best_model:
                with open("best_model.pkl", "wb") as f:
                    pickle.dump(best_model, f)
                st.success(f"🏅 Best Model: {max(results, key=lambda k: results[k]['Accuracy'])} with Accuracy: {best_acc:.2f}")
                with open("best_model.pkl", "rb") as f:
                    st.download_button("⬇️ Download Best Model (Pickle)", f, file_name="best_model.pkl")

        try:
            fig, ax = plt.subplots()
            pd.DataFrame(results).T[['Accuracy', 'AUC']].plot(kind='bar', ax=ax)
            plt.title("Model Comparison")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Plotting error: {e}")

        st.write("### 🧾 Classification Report & Confusion Matrix")
        selected_model_name = st.selectbox("Select a model to inspect", list(results.keys()))

        if selected_model_name:
            model = models[selected_model_name]
            y_pred = model.predict(X_test_preprocessed)

            st.subheader("📋 Classification Report")
            st.text(classification_report(y_test, y_pred))

            st.subheader("🔢 Confusion Matrix")
            fig_cm, ax_cm = plt.subplots()
            cm = confusion_matrix(y_test, y_pred)
            im = ax_cm.imshow(cm, cmap='Blues')
            plt.colorbar(im)
            ax_cm.set_title("Confusion Matrix")
            st.pyplot(fig_cm)

