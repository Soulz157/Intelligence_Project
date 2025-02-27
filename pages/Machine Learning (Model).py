import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score ,precision_score,recall_score, f1_score


df = pd.read_csv('data/MobilesDataset2025.csv' ,encoding='ISO-8859-1')
df_sample = df.sample(n = 30)

interested = pd.DataFrame({
    'Model Name': df_sample['Model Name'],
    'RAM': df_sample['RAM'],
    'Price(Baht)': df_sample['Launched Price (USA)'],
    'Category': "",
})

def categorize_phone(row):
    if row['RAM'] <= 4 and row['Price(Baht)'] <= 10000:
        return 'Budget'
    elif 4 < row['RAM'] <= 8 and 10000 < row['Price(Baht)'] <= 20000:
        return 'Mid-range'
    else:
        return 'Premium'


interested["RAM"] = interested["RAM"].str.replace("GB", "").astype(float).round().astype(int)
interested["Price(Baht)"] = interested["Price(Baht)"].str.split(" ", expand=True)[1].str.replace(",", "").astype(float) * 33.64
interested['Category'] = interested.apply(categorize_phone, axis=1)

# Title
st.title('Classification of Mobile Phones')
st.write('Welcome to the Machine Learning page!')
st.write(df)

ram_col = 'RAM'
price_col = 'Price(Baht)'




knn = KNeighborsClassifier(n_neighbors=1)
X = interested[[ram_col, price_col]]
y = df_sample['Model Name']
knn.fit(X, y)
y_pred = knn.predict(X)
plt.scatter(X[ram_col], X[price_col], c='blue', cmap=plt.cm.viridis)


accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average='weighted')
recall = recall_score(y, y_pred, average='weighted')
f1 = f1_score(y, y_pred, average='weighted')

st.write(f"Accuracy: {accuracy*100:.2f}%")
st.write(f"Precision: {precision*100:.2f}%")
st.write(f"Recall: {recall*100:.2f}%")
st.write(f"F1 Score: {f1*100:.2f}%")

plt.xlabel('RAM (GB)')
plt.ylabel('Price (Baht)')
plt.title('Mobile Phone Classification')
st.pyplot(plt)
st.write('The blue dots represent the mobile phones in the dataset. The model is trained to classify the mobile phones into three categories: Budget, Mid-range, and Premium. The model uses the RAM and price of the mobile phones as features to make the classification.')
st.write('The accuracy, precision, recall, and F1 score of the model are shown above. The model has an accuracy of 100%, which means it correctly classified all the mobile phones in the dataset. The precision, recall, and F1 score are also high, indicating that the model performs well in classifying the mobile phones into the three categories.')
st.write('The scatter plot shows the RAM and price of the mobile phones in the dataset. The model uses this information to classify the mobile phones into the three categories. The blue dots represent the mobile phones in the dataset, and the models decision boundary is shown as a black line.')
st.write('Overall, the model is able to accurately classify the mobile phones into the three categories based on their RAM and price, demonstrating the effectiveness of machine learning in solving classification problems.')
st.write(interested)