import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Judul dashboard
st.title("Dashboard Analisis Kualitas Wine")

# Load data
df = pd.read_csv("WineQT.csv")

# Labeling target
df['quality'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

# Visualisasi distribusi target
st.subheader("Distribusi Kualitas Wine (0 = Biasa, 1 = Bagus)")
fig1, ax1 = plt.subplots()
sns.countplot(x='quality', data=df, palette=['red', 'green'], ax=ax1)
st.pyplot(fig1)

# Heatmap korelasi
st.subheader("Korelasi antar Fitur")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax2)
st.pyplot(fig2)

# Training model
X = df.drop(columns='quality')
y = df['quality']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediksi & evaluasi
y_pred = model.predict(X_test)
st.subheader("Confusion Matrix")
fig3, ax3 = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax3)
st.pyplot(fig3)

# Classification report
st.subheader("Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())
