import streamlit as st
import numpy as np
import pandas as pd
import pathlib


def load_css(file_name: str) -> None:
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


css_path = pathlib.Path("css/style.css")
load_css(css_path)


df = pd.read_csv('data/MobilesDataset2025.csv', encoding='ISO-8859-1')

st.title('📞 Classification of Mobile Phones')
st.markdown('''<div class="container">
            <div class="row">
                <div class="col-contentML">
                    <h3>• ที่มาของData Set</h3>
                    <p>ผมได้นำDataset มา จาก
                    <a href="https://www.kaggle.com/datasets/abdulmalik1518/mobiles-dataset-2025"> Kaggle </a> 
                    ซึ่งเป็น Dataset ของ Moblieใน ปี2024 โดยที่ตัว Dataset จะมี Weight Processor Ram Front/Back camera BatteryCapacity Priceในสกุลเงินต่างๆ และ Screen size เป็น feature เพื่อมาจำแนกและแบ่งประเภทของ Mobile ออกเป็น 3 ระดับ: Budget (ราคาประหยัด), Mid-range (ระดับกลาง), Premium (ระดับพรีเมียม) โดยใช้ข้อมูลของ RAM และ Price ในการทำนาย
                    โดย Algorithm ที่ผมเลือกที่จะนำมาใช้คือ K-Nearest Neighbors (KNN) และ Support Vector Machine (SVM)</p>
                </div>
            </div>
        </div>''', unsafe_allow_html=True)

st.divider()
st.markdown("<h3 style='text-align: center; color: #00000;'>Raw Data</h3>",
            unsafe_allow_html=True)
st.write(df)

st.divider()
st.subheader('Company Name Filter From Raw Data')
st.markdown(
    f"""
    <div style="display: flex; flex-wrap: nowrap; overflow-x: auto;">
        {' '.join([f'<div style="padding: 10px; margin-right: 20px;">{name}</div>' for name in df['Company Name'].unique()])}
    </div>
    """, unsafe_allow_html=True
)

st.divider()
st.subheader('ทฤษฎีของ K-Nearest Neighbors (KNN)')
st.write("• K-Nearest Neighbors (KNN)เป็น วิธีการแบ่งคลาสเพื่อไว้สำหรับจัดหมวดหมู่ข้อมูล(Classification) โดยใช้ชุดข้อมูลที่ใกล้ที่สุด(K) ซึ่งคำนวณจากระยะห่างระหว่างจุดข้อมูล กับ ข้อมูลใหม่ที่จะทำนาย เพื่อที่จะใช้ค่าเฉลี่ยของคลาสของข้อมูลที่ใกล้เคียงที่สุดเพื่อทำนายคลาสของข้อมูลใหม่")

st.divider()
st.subheader("ขั้นตอนการพัฒนาโมเดล K-Nearest Neighbors (KNN)")
st.subheader("- Data Cleansing")
st.write("หลังจากที่เราทำการอ่านข้อมูลเข้ามาแล้ว เราก็จะสร้างDataframeเพื่อที่จะเก็บข้อมูลที่เราจะใช้ในการทำนายและได้ทำการเปลี่ยนค่าเงินให้เป็นค่าเงินของประเภทไทยและผมก็ได้ทำการแบ่งcategoryไว้คร่าวๆก่อนเพื่อที่จะนำไปtrainโมเดล")    
st.code('''
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
        
interested['Price(Baht)'] = interested['Price(Baht)'].astype(str).str.split(" ", expand=True)[1].str.replace(',', '', regex=True).astype(float) * 33.64
interested['Price(Baht)'] = interested['Price(Baht)'].fillna(interested['Price(Baht)'].median())
        
category_mapping = {'Budget': 0, 'Mid-range': 1, 'Premium': 2}
interested['Category'] = interested.apply(categorize_phone, axis=1).map(
    category_mapping).fillna(0).astype(int)
        ''')
st.divider()
st.write("ต่อมาผมก็ได้สร้างModel KNN ขึ้นมาโดยใช้ค่า K เป็น 5 และแบ่งข้อมูลออกเป็น X(feature) และ Y(Label) และทำการ trainโมเดลด้วยชุดข้อมูลที่มี")
st.code(''' 
    knn = KNeighborsClassifier(n_neighbors=5)
    X = interested[[ram_col, price_col]]
    y = interested['Category']
        
    X_predict = interested[[ram_col, price_col]]
    knn.fit(X, y)
    y_pred = knn.predict(X_predict))
        ''')

st.divider()
st.write("ซึ่งผมได้ทำการPlotกราฟScatter plotออกมาเพื่อดูว่าโมเดลมีการแบ่งClassificationsอย่างไรโดยใชสีเป็นพื้นหลังเพื่อให้ดูง่ายและกำหนดscaleสีของcolorbarเพื่อให้เห็นว่าแต่ละสีคือClassificationsอะไร")
st.code(''' 
    x_min, x_max = X[ram_col].min() - 1, X[ram_col].max() + 1
    y_min, y_max = X[price_col].min() - 5000, X[price_col].max() + 5000
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),np.linspace(y_min, y_max, 100))

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.viridis)

    scatter = plt.scatter(X[ram_col], X[price_col], c=y,cmap=plt.cm.viridis, edgecolor="k")
    plt.colorbar(scatter, ticks=[0, 1, 2], label="Category (Budget(0) Mid-range(1) Premium(2))")

    y_min, y_max = X[price_col].min(), X[price_col].max()
    plt.yticks(np.arange(y_min, y_max + 5000, y_max//10))
        
    plt.xlabel('RAM (GB)')
    plt.ylabel('Price (Baht)')
    plt.title('Mobile Phone Classification')
    st.pyplot(plt)
''')
st.divider()
st.write("หลังจากนั้นผมก็มาคำนวณ Accuracy, Precision, Recall, F1 Score และ Confusion Matrixของโมเดล")
st.code('''
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
        
    st.write(f"Accuracy: {accuracy*100:.2f}%")
    st.write(f"Precision: {precision*100:.2f}%")
    st.write(f"Recall: {recall*100:.2f}%")
    st.write(f"F1 Score: {f1*100:.2f}%")
        
    cm = confusion_matrix(y, y_pred)
    st.subheader("Confusion Matrix : ")
    st.write(pd.DataFrame(cm, columns=[f'Predicted {i}' for i in range( cm.shape[1])], index=[f'Actual {i}' for i in range(cm.shape[0])]))
    ''')

st.divider()
st.subheader('ทฤษฎีของ Support Vector Machine (SVM)')
st.write("• Support Vector Machine (SVM) เป็น วิธีการแบ่งคลาสเพื่อไว้สำหรับจัดหมวดหมู่ข้อมูล(Classification) โดยจะสร้างเส้นแบ่งคลาสมาเพื่อแยกประเภทของข้อมูลและจะเลือกใช้เส้นแบ่งคลาสที่ดีที่สุด โดยที่เส้นแบ่งคลาสนั้นจะมีระยะห่างระหว่างข้อมูลที่ใกล้ที่สุดมากที่สุดเพื่อมาทำนาย")

st.divider()
st.subheader("ขั้นตอนการพัฒนาโมเดล Support Vector Machine (SVM)")
st.subheader("- Data Cleansing")
st.write("หลังจากที่เราทำการอ่านข้อมูลเข้ามาแล้วซึ่งในส่วนนี้จะเก็บเหมือนกับ KNN และเราก็จะสร้างDataframeเพื่อที่จะเก็บข้อมูลที่เราจะใช้ในการทำนายและได้ทำการเปลี่ยนค่าเงินให้เป็นค่าเงินของประเภทไทยและผมก็ได้ทำการแบ่งcategoryไว้คร่าวๆก่อนเพื่อที่จะนำไปtrainโมเดล")    
st.code('''
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
        
interested['Price(Baht)'] = interested['Price(Baht)'].astype(str).str.split(" ", expand=True)[1].str.replace(',', '', regex=True).astype(float) * 33.64
interested['Price(Baht)'] = interested['Price(Baht)'].fillna(interested['Price(Baht)'].median())
        
category_mapping = {'Budget': 0, 'Mid-range': 1, 'Premium': 2}
interested['Category'] = interested.apply(categorize_phone, axis=1).map(
    category_mapping).fillna(0).astype(int)
        ''')
st.divider()
st.write("ต่อมาผมก็ได้ทำการสร้างModel SVM ขึ้นมาโดยใช้kernelเป็นlinearและแบ่งข้อมูลออกเป็น X(feature) และ Y(Label) หลังจากนั้นก็แบ่งข้อมูลทั้ง X และ Y ออกเป็น ชุดTrain และ ชุดTest และนำไปtrainโมเดล")
st.code(''' 
   svm = SVC(kernel='linear', C=0.3)
    X = interested[[ram_col, price_col]]
    y = interested['Category']
        
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.80, random_state=101)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
        ''')
st.divider()
st.write("และผมได้ทำการPlotกราฟScatter plotออกมาเพื่อดูว่าโมเดลมีการแบ่งClassificationsอย่างไรโดยใชสีเป็นพื้นหลังเพื่อให้ดูง่ายและกำหนดscaleสีของcolorbarเพื่อให้เห็นว่าแต่ละสีคือClassificationsอะไร")
st.code('''
    x_min, x_max = X[ram_col].min() - 1, X[ram_col].max() + 1
    y_min, y_max = X[price_col].min() - 5000, X[price_col].max() + 5000
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),np.linspace(y_min, y_max, 100))

    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.viridis)

    plot = plt.subplot()
    scatter = plt.scatter(X[ram_col], X[price_col], c=y,cmap=plt.cm.viridis, edgecolor="k")
    plt.colorbar(scatter, ticks=[0, 1, 2],label="Category (Budget(0) Mid-range(1) Premium(2))")

    plt.xlabel('RAM (GB)')
    plt.ylabel('Price (Baht)')
    plt.title('Mobile Phone Classification')
    st.pyplot(plt)
        ''')

st.divider()
st.write("ทำการคำนวณค่า Accuracy, Precision, Recall, F1 Score และ Confusion Matrixของโมเดล")
st.code('''
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
    st.write(pd.DataFrame(cm, columns=[f'Predicted {i}' for i in range(cm.shape[1])], index=[f'Actual {i}' for i in range(cm.shape[0])]))
''')
st.divider()
st.subheader("สามารถดูโมเดลได้โดยการกดที่ด้านล่าง")
st.page_link("./pages/machine_learning_model",label="Machine Learning Model",icon="🧠")

st.subheader("-หรือสามารถดูโค้ดได้ที่Github-")
st.link_button("🔗 Source Code", "https://github.com/Soulz157/Intelligence_Project")