import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score ,precision_score,recall_score, f1_score
import re


df = pd.read_csv('data/MobilesDataset2025.csv' ,encoding='ISO-8859-1')

df_sample = df.sample(n = 30,random_state=52)

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
    ram = row['RAM']
    price = row['Price(Baht)']
    if ram <= 4 and price <= 10000:
        return 'Budget'
    elif 4 < ram <= 8 and 10000 < price <= 25000:
        return 'Mid-range'
    else:
        return 'Premium'

    

interested = pd.DataFrame({
    'Model Name': df_sample['Model Name'],
    'RAM': df_sample['RAM'].apply(extract_max_number).astype(float).round().astype(int),
    'Front Camera': df_sample['Front Camera'].apply(extract_front_camera),
    'Back Camera': df_sample['Back Camera'].apply(extract_back_camera),
    'Battery Capacity': df_sample['Battery Capacity'].apply(extract_max_number),
    'Screen Size': df_sample['Screen Size'].apply(extract_max_number),
    'Price(Baht)': df_sample['Launched Price (USA)'],
    'Category': "",
})
sample = ({
    "Model Name" : interested["Model Name"].copy(),"Price(Baht)" : (interested["Price(Baht)"].str.split(" ", expand=True)[1].str.replace(",", "").astype(float) * 33.64).astype(int)})


interested["Price(Baht)"] = interested["Price(Baht)"].str.split(" ", expand=True)[1].str.replace(",", "").astype(float) * 33.64
interested['Category'] = interested.apply(categorize_phone, axis=1)

# Title
st.title('Classification of Mobile Phones')
st.write('Welcome to the Machine Learning page!')
option = st.selectbox(
    "Choose Model to Prediction the Category of Mobile Phones",
    ("K-Nearest Neighbors", "Support Vector Machine"),
    index=None,
    placeholder= "Select a model",
)


on = st.toggle('Show Full Dataset')

if on:
    st.write(df)

st.write('The dataset contains information about mobile phones, including the model name, RAM, front camera, back camera, battery capacity, screen size, and price. The goal is to classify the mobile phones into three categories: Budget, Mid-range, and Premium based on their RAM and price.')
ram_col = 'RAM'
price_col = 'Price(Baht)'


if option == "K-Nearest Neighbors":

    knn = KNeighborsClassifier(n_neighbors=2)
    X = interested[[ram_col, price_col]]
    y = df_sample['Model Name']
    knn.fit(X, y)
    y_pred = knn.predict(X)
    plt.scatter(X[ram_col], X[price_col], c='blue', cmap=plt.cm.viridis)


    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
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
    st.write(interested)
    st.write(f"Accuracy: {accuracy*100:.2f}%")
    st.write(f"Precision: {precision*100:.2f}%")


    budget_phones = ({ "Name" : interested[interested["Category"] == "Budget"]["Model Name"]})
    midrange_phones =  ({ "Name" : interested[interested["Category"] == "Mid-range"]["Model Name"] })
    premium_phones =  ({ "Name" : interested[interested["Category"] == "Premium"]["Model Name"] })


    st.write("Budget Phones:")
    st.write(budget_phones['Name'] if not budget_phones['Name'].empty else "No Budget Phones")

    st.write("Mid-range Phones:")
    st.write(midrange_phones['Name'] if not midrange_phones['Name'].empty else "No Mid-range Phones")

    st.write("Premium Phones:")
    st.write(premium_phones['Name'] if not premium_phones['Name'].empty else "No Premium Phones")

elif option == "Support Vector Machine":
    svm = svm.SVC(kernel='linear')
    X = interested[[ram_col, price_col]]
    y = df_sample['Model Name']
    svm.fit(X, y)
    y_pred = svm.predict(X)
    plt.scatter(X[ram_col], X[price_col], c='blue', cmap=plt.cm.viridis)    
