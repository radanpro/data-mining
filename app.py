import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Log steps to an .md file
def log_step(step_name, description):
    with open("documentation.md", "a", encoding="utf-8") as doc_file:
        doc_file.write(f"## {step_name}\n{description}\n\n")

# Save plots to images
def save_plot(fig, filename):
    if not os.path.exists("plots"):
        os.makedirs("plots")
    fig.savefig(f"plots/{filename}.png")
    log_step("Save Plot", f"Plot saved as {filename}.png")

# Load data
@st.cache_data
def load_data(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)
        log_step("Load Data", "Data successfully loaded.")
        return data
    except Exception as e:
        log_step("Load Data", f"Error loading data: {e}")
        st.error("Failed to load data.")
        return None

# Main application
def main():
    st.title("Advanced Data Mining Pipeline")
    st.write("Comprehensive data analysis and machine learning platform")
    
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file:
        data = load_data(uploaded_file)
        
        if data is not None:
            st.subheader("Raw Data Preview")
            st.dataframe(data.head())
            st.write(f"Dataset Shape: {data.shape}")
            
            # Apriori Section
            st.subheader("Step 1: Association Rule Mining (Apriori)")
            st.write("### Preprocessing for Apriori")
            
            # تحويل البيانات إلى ثنائية (0/1)
            apriori_data = data.copy()
            categorical_cols = apriori_data.select_dtypes(include=["object"]).columns
            if len(categorical_cols) > 0:
                apriori_data = pd.get_dummies(apriori_data, columns=categorical_cols, drop_first=True)
            apriori_data = apriori_data.applymap(lambda x: 1 if x > 0 else 0)
            
            st.write("### Processed Data for Apriori")
            st.dataframe(apriori_data.head())
            
            min_support = st.slider("Minimum Support", 0.01, 0.5, 0.1, key="apriori_support")
            min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.7, key="apriori_confidence")
            
            if st.button("Run Apriori"):
                try:
                    frequent_itemsets = apriori(apriori_data, min_support=min_support, use_colnames=True)
                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
                    
                    st.write("### Results")
                    st.dataframe(rules.sort_values("lift", ascending=False))
                    
                    st.write("### Business Insights")
                    best_rule = rules.iloc[0]
                    st.success(f"**Key Insight**: When customers buy {best_rule['antecedents']}, " 
                              f"they are {best_rule['confidence']*100:.1f}% likely to also buy {best_rule['consequents']} "
                              f"(Lift = {best_rule['lift']:.2f})")
                except Exception as e:
                    st.error(f"Error in Apriori: {str(e)}")
            
            # Naive Bayes Section
            st.subheader("Step 2: Classification (Naive Bayes)")
            st.write("### Preprocessing for Naive Bayes")
            
            nb_data = data.copy()
            numerical_cols = nb_data.select_dtypes(include=[np.number]).columns
            categorical_cols = nb_data.select_dtypes(include=["object"]).columns
            
            # Handle missing values
            imputer = SimpleImputer(strategy="mean")
            nb_data[numerical_cols] = imputer.fit_transform(nb_data[numerical_cols])
            
            # Encode categorical columns
            label_encoders = {}
            for col in categorical_cols:
                label_encoders[col] = LabelEncoder()
                nb_data[col] = label_encoders[col].fit_transform(nb_data[col])
            
            st.write("### Processed Data for Naive Bayes")
            st.dataframe(nb_data.head())
            
            target_col_nb = st.selectbox("Select Target Variable", nb_data.columns, key="nb_target")
            
            if st.button("Run Naive Bayes"):
                X = nb_data.drop(columns=[target_col_nb])
                y = nb_data[target_col_nb]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = GaussianNB()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                acc = accuracy_score(y_test, predictions)
                cm = confusion_matrix(y_test, predictions)
                
                st.write("### Results")
                st.write(f"**Accuracy**: {acc:.2%}")
                
                st.write("### Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
            
            # Decision Tree Section
            st.subheader("Step 3: Decision Tree (ID3)")
            st.write("### Preprocessing for Decision Tree")
            
            dt_data = data.copy()
            numerical_cols = dt_data.select_dtypes(include=[np.number]).columns
            categorical_cols = dt_data.select_dtypes(include=["object"]).columns
            
            # Handle missing values
            imputer = SimpleImputer(strategy="mean")
            dt_data[numerical_cols] = imputer.fit_transform(dt_data[numerical_cols])
            
            # Encode categorical columns
            label_encoders = {}
            for col in categorical_cols:
                label_encoders[col] = LabelEncoder()
                dt_data[col] = label_encoders[col].fit_transform(dt_data[col])
            
            st.write("### Processed Data for Decision Tree")
            st.dataframe(dt_data.head())
            
            target_col_dt = st.selectbox("Select Target Variable", dt_data.columns, key="dt_target")
            max_depth = st.slider("Max Tree Depth", 1, 10, 3, key="tree_depth")
            
            if st.button("Run Decision Tree"):
                X = dt_data.drop(columns=[target_col_dt])
                y = dt_data[target_col_dt]
                model = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, random_state=42)
                model.fit(X, y)
                
                st.write("### Decision Rules")
                fig, ax = plt.subplots(figsize=(20, 10))
                plot_tree(model, filled=True, feature_names=X.columns, class_names=[str(c) for c in y.unique()], ax=ax)
                st.pyplot(fig)
                
                st.write("### Feature Importance")
                importance = pd.Series(model.feature_importances_, index=X.columns)
                st.bar_chart(importance.sort_values(ascending=False))
            
            # K-Means Section
            st.subheader("Step 4: Clustering (K-Means)")
            st.write("### Preprocessing for K-Means")
            
            kmeans_data = data.copy()
            numerical_cols = kmeans_data.select_dtypes(include=[np.number]).columns
            categorical_cols = kmeans_data.select_dtypes(include=["object"]).columns
            
            # Handle missing values
            imputer = SimpleImputer(strategy="mean")
            kmeans_data[numerical_cols] = imputer.fit_transform(kmeans_data[numerical_cols])
            
            # Encode categorical columns
            label_encoders = {}
            for col in categorical_cols:
                label_encoders[col] = LabelEncoder()
                kmeans_data[col] = label_encoders[col].fit_transform(kmeans_data[col])
            
            st.write("### Processed Data for K-Means")
            st.dataframe(kmeans_data.head())
            
            n_clusters = st.slider("Number of Clusters", 2, 10, 3, key="kmeans_clusters")
            
            if st.button("Run K-Means"):
                model = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = model.fit_predict(kmeans_data)
                score = silhouette_score(kmeans_data, clusters)
                
                st.write("### Results")
                st.write(f"**Silhouette Score**: {score:.2f}")
                
                st.write("### Cluster Distribution")
                cluster_counts = pd.Series(clusters).value_counts().sort_index()
                st.bar_chart(cluster_counts)
                
                st.write("### Cluster Visualization")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x=kmeans_data.iloc[:, 0], y=kmeans_data.iloc[:, 1], hue=clusters, palette="viridis", ax=ax)
                ax.set_title("Customer Segmentation")
                st.pyplot(fig)

if __name__ == "__main__":
    main()