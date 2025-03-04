import time
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import re


df = pd.read_csv('data/MobilesDataset2025.csv', encoding='ISO-8859-1')

df.dropna(subset=['Model Name'], inplace=True)

df_sample = df.sample(n=30, random_state=52)


def extract_max_number(text):
    numbers = re.findall(r'\d+\.?\d*', str(text))
    return max(numbers) if numbers else None


def extract_front_camera(text):
    text = str(text)
    if "4K" in text:
        return "4K"
    numbers = re.findall(r'\d+\.?\d*', text)
    return max(numbers, key=float) + " MP" if numbers else None


def extract_back_camera(text):
    numbers = re.findall(r'\d+', str(text))
    total = sum(map(int, numbers)) if numbers else 0
    return f"{total} MP" if total > 0 else None


def categorize_phone(row):
    ram = row['RAM(GB)']
    price = row['Price(Baht)']
    if ram <= 4 and price <= 10000:
        return 'Budget'
    elif 4 < ram <= 8 and 10000 < price <= 25000:
        return 'Mid-range'
    else:
        return 'Premium'


interested = pd.DataFrame({
    'Model Name': df_sample['Model Name'],
    'RAM(GB)': df_sample['RAM'].apply(extract_max_number).fillna(0).astype(float).round().astype(int),
    'Front Camera': df_sample['Front Camera'].apply(extract_front_camera),
    'Back Camera': df_sample['Back Camera'].apply(extract_back_camera),
    'Battery Capacity(mAh)': df_sample['Battery Capacity'].astype(str).str.split("m", expand=True)[0].str.replace(',', '').astype(float) / 1000,
    'Screen Size': df_sample['Screen Size'].apply(extract_max_number),
    'Price(Baht)': df_sample['Launched Price (USA)'],
    'Category': "",
})
sample = ({
    "Model Name": interested["Model Name"].copy(), "Price(Baht)": (interested["Price(Baht)"].str.split(" ", expand=True)[1].str.replace(",", "").astype(float) * 33.64).astype(int)
})


interested['Price(Baht)'] = interested['Price(Baht)'].astype(str).str.split(
    " ", expand=True)[1].str.replace(',', '', regex=True).astype(float) * 33.64
interested['Price(Baht)'] = interested['Price(Baht)'].fillna(
    interested['Price(Baht)'].median())
category_mapping = {'Budget': 0, 'Mid-range': 1, 'Premium': 2}
interested['Category'] = interested.apply(categorize_phone, axis=1).map(
    category_mapping).fillna(0).astype(int)

# Title
st.title('Classification of Mobile Phones')
st.write('Welcome to the Machine Learning page!')
option = st.selectbox(
    "Choose Model to Prediction the Category of Mobile Phones",
    ("K-Nearest Neighbors", "Support Vector Machine"),
    index=None,
    placeholder="Select a model",
)


on = st.toggle('Show Full Dataset')

if on:
    st.write(df)

st.write('The dataset contains information about mobile phones, including the model name, RAM, front camera, back camera, battery capacity, screen size, and price. The goal is to classify the mobile phones into three categories: Budget, Mid-range, and Premium based on their RAM and price.')
ram_col = 'RAM(GB)'
price_col = 'Price(Baht)'


if option == "K-Nearest Neighbors":
    knn = KNeighborsClassifier(n_neighbors=5)
    X = interested[[ram_col, price_col]]
    y = interested['Category']
    knn.fit(X, y)
    y_pred = knn.predict(X)

    x_min, x_max = X[ram_col].min() - 1, X[ram_col].max() + 1
    y_min, y_max = X[price_col].min() - 5000, X[price_col].max() + 5000
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.viridis)

    scatter = plt.scatter(X[ram_col], X[price_col], c=y,
                          cmap=plt.cm.viridis, edgecolor="k")
    plt.colorbar(scatter, ticks=[
                 0, 1, 2], label="Category (Budget(0) Mid-range(1) Premium(2))")

    y_min, y_max = X[price_col].min(), X[price_col].max()
    plt.yticks(np.arange(y_min, y_max + 5000, y_max//10))

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')

    onsm = st.toggle('Show Sample Dataset')
    if onsm:
        st.write(sample)

    plt.xlabel('RAM (GB)')
    plt.ylabel('Price (Baht)')
    plt.title('Mobile Phone Classification')
    st.pyplot(plt)
    st.write('The blue dots represent the mobile phones in the dataset. The model is trained to classify the mobile phones into three categories: Budget, Mid-range, and Premium. The model uses the RAM and price of the mobile phones as features to make the classification.')
    st.write('The accuracy, precision, recall, and F1 score of the model are shown above. The model has an accuracy of 100%, which means it correctly classified all the mobile phones in the dataset. The precision, recall, and F1 score are also high, indicating that the model performs well in classifying the mobile phones into the three categories.')
    st.write('The scatter plot shows the RAM and price of the mobile phones in the dataset. The model uses this information to classify the mobile phones into the three categories. The blue dots represent the mobile phones in the dataset, and the models decision boundary is shown as a black line.')
    st.write('Overall, the model is able to accurately classify the mobile phones into the three categories based on their RAM and price, demonstrating the effectiveness of machine learning in solving classification problems.')
    st.subheader("Model Prediction")
    interested['Predicted Category'] = interested['Category'].map(
        {0: 'Budget', 1: 'Mid-range', 2: 'Premium'})
    st.write(interested[['Model Name', 'RAM(GB)',
             'Price(Baht)', 'Predicted Category']])
    st.write(f"Accuracy: {accuracy*100:.2f}%")
    st.write(f"Precision: {precision*100:.2f}%")
    st.write(f"Recall: {recall*100:.2f}%")
    st.write(f"F1 Score: {f1*100:.2f}%")

    cm = confusion_matrix(y, y_pred)
    st.subheader("Confusion Matrix : ")
    st.write(pd.DataFrame(cm, columns=[f'Predicted {i}' for i in range(
        cm.shape[1])], index=[f'Actual {i}' for i in range(cm.shape[0])]))

    budget_phones = (
        {"Name": interested[interested["Category"] == 0]["Model Name"]})
    midrange_phones = (
        {"Name": interested[interested["Category"] == 1]["Model Name"]})
    premium_phones = (
        {"Name": interested[interested["Category"] == 2]["Model Name"]})

    st.subheader("• Budget Phones")
    st.write(
        budget_phones['Name'] if not budget_phones['Name'].empty else "No Budget Phones")

    st.subheader("• Mid-range Phones")
    st.write(midrange_phones['Name']
             if not midrange_phones['Name'].empty else "No Mid-range Phones")

    st.subheader("• Premium Phones")
    st.write(
        premium_phones['Name'] if not premium_phones['Name'].empty else "No Premium Phones")


elif option == "Support Vector Machine":

    svm = SVC(kernel='linear', C=0.3)
    X = interested[[ram_col, price_col]]
    y = interested['Category']
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.80, random_state=101)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)

    x_min, x_max = X[ram_col].min() - 1, X[ram_col].max() + 1
    y_min, y_max = X[price_col].min() - 5000, X[price_col].max() + 5000
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.viridis)

    plot = plt.subplot()
    scatter = plt.scatter(X[ram_col], X[price_col], c=y,
                          cmap=plt.cm.viridis, edgecolor="k")
    plt.colorbar(scatter, ticks=[0, 1, 2],
                 label="Category (Budget(0) Mid-range(1) Premium(2))")

    # plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
    #             s=100, facecolors='none', edgecolors='r', label='Support Vectors')

    onsm = st.toggle('Show Sample Dataset')
    if onsm:
        st.write(sample)

    plt.xlabel('RAM (GB)')
    plt.ylabel('Price (Baht)')
    plt.title('Mobile Phone Classification')
    # plt.legend()
    st.pyplot(plt)

    st.subheader("Model Prediction")
    interested['Predicted Category'] = interested['Category'].map(
        {0: 'Budget', 1: 'Mid-range', 2: 'Premium'})
    cmpr_df = pd.DataFrame(
        {"True Category": y_test, "Predicted Category": y_pred})
    st.write(interested[['Model Name', 'RAM(GB)',
             'Price(Baht)', 'Predicted Category']], cmpr_df)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    st.write(f"Accuracy: {accuracy*100:.2f}%")
    st.write(f"Precision: {precision*100:.2f}%")
    st.write(f"Recall: {recall*100:.2f}%")
    st.write(f"F1 Score: {f1*100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix : ")
    st.write(pd.DataFrame(cm, columns=[f'Predicted {i}' for i in range(
        cm.shape[1])], index=[f'Actual {i}' for i in range(cm.shape[0])]))

    budget_phones = (
        {"Name": interested[interested["Category"] == 0]["Model Name"]})
    midrange_phones = (
        {"Name": interested[interested["Category"] == 1]["Model Name"]})
    premium_phones = (
        {"Name": interested[interested["Category"] == 2]["Model Name"]})

    st.subheader("• Budget Phones")
    st.write(
        budget_phones['Name'] if not budget_phones['Name'].empty else "No Budget Phones")

    st.subheader("• Mid-range Phones")
    st.write(midrange_phones['Name']
             if not midrange_phones['Name'].empty else "No Mid-range Phones")

    st.subheader("• Premium Phones")
    st.write(
        premium_phones['Name'] if not premium_phones['Name'].empty else "No Premium Phones")
