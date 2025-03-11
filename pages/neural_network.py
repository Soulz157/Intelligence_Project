import streamlit as st
import numpy as np
import pathlib
import pandas as pd
import matplotlib.pyplot as plt


def load_css(file_name: str) -> None:
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


css_path = pathlib.Path("css/style.css")
load_css(css_path)

df = pd.read_csv('data/train.csv')


st.title('🎵 Music Genre Classification')
st.markdown('''<div class="container">
            <div class="row">
                <div class="col-contentNTML">
                    <h3>• ที่มาของData Set</h3>
                    <p>Datasetตัวนี้ผมนำ มาจาก
                    <a href="https://www.kaggle.com/datasets/thedevastator/spotify-tracks-genre-dataset"> Kaggle </a> 
                    โดยผมอยากที่จะทราบว่าศิลปินชอบที่จะทำเพลงแนวไหนสไตล์ไหนผมเลยได้นำDatasetชุดนี้มาทำการPredictศิลปินว่าศิลปินคนนี้ชอบทำเพลงแนวไหนมาทดสอบกับ Neural Network โดยใช้Model Feedforward Neural Networks (FNN)</p>
                </div>
            </div>
        </div>''', unsafe_allow_html=True)

st.subheader("Feature of Data")
st.markdown('''<div class="container"
                    <div class="row">
         <ul>
            <li><b>artists:</b> ชื่อของศิลปินที่เกี่ยวข้องกับเพลง</li>
            <li><b>album_name:</b> ชื่ออัลบั้มที่เพลงนั้นอยู่</li>
            <li><b>track_name:</b> ชื่อเพลง</li>
            <li><b>popularity:</b> คะแนนความนิยมของเพลงบน Spotify (0-100)</li>
            <li><b>duration_ms:</b> ความยาวของเพลง (มิลลิวินาที)</li>
            <li><b>explicit:</b> เพลงมีเนื้อหาที่ไม่เหมาะสมหรือไม่</li>
            <li><b>danceability:</b> ความเหมาะสมของเพลงสำหรับการเต้น (0-1)</li>
            <li><b>energy:</b> ความเข้มข้นและพลังของเพลง (0-1)</li>
            <li><b>key:</b> คีย์ของเพลง (ตัวเลข)</li>
            <li><b>loudness:</b> ระดับความดังของเพลง (เดซิเบล)</li>
            <li><b>mode:</b> โหมดของเพลง (0 = ไมเนอร์, 1 = เมเจอร์)</li>
            <li><b>speechiness:</b> ระดับของเสียงพูดในเพลง (0-1)</li>
            <li><b>acousticness:</b> ความเป็นอะคูสติกของเพลง (0-1)</li>
            <li><b>instrumentalness:</b> โอกาสที่เพลงจะไม่มีเสียงร้อง (0-1)</li>
            <li><b>liveness:</b> การบ่งบอกว่ามีผู้ฟังสดขณะบันทึกเพลง (0-1)</li>
            <li><b>valence:</b> อารมณ์เชิงบวกของเพลง (0-1)</li>
            <li><b>tempo:</b> ความเร็วของเพลง (BPM)</li>
            <li><b>time_signature:</b> จำนวนจังหวะในแต่ละห้องเพลง</li>
            <li><b>track_genre:</b> แนวเพลงของเพลง</li>
        </ul>
                    </div>
            </div>''', unsafe_allow_html=True)

st.divider()
st.markdown("<h3 style='text-align: center; color: #00000;'>Raw Data</h3>",
            unsafe_allow_html=True)
st.write(df)
st.divider()

st.subheader("ทฤษฎีของ Feedforward Neural Networks (FNN)")
st.write("• Feedforward Neural Networks (FNN) คือ โมเดลประสาทเทียม(Neural Networks) ที่มีโครงสร้างประกอบด้วย Layer ของ Nodeที่เชื่อมต่อกันเป็นลำดับโดยการเชื่อมต่อของโหนดแต่ละชั้นเป็นเส้นตรง และข้อมูลจะถูกส่งผ่านจากลำดับแรกไปยังลำดับถัดไปจนกระทั่งลำดับสุดท้ายที่จะได้ผลลัพธ์เป็นoutput ออกมา โดยไม่มีการส่งข้อมูลมาชั้นก่อนหน้า")
st.write("- โครงสร้างของ FNN ประกอบด้วย 3 ชั้นหลัก คือ Input Layer, Hidden Layer และ Output Layer")
st.divider()

st.subheader("ขั้นตอนการพํฒนาโมเดล Feedforward Neural Networks (FNN)")
st.subheader("- Data Cleansing")
st.write("หลังจากที่ทำการโหลดข้อมูลมาแล้ว จะต้องทำการตรวจสอบข้อมูลว่ามีข้อมูลที่หายไปหรือไม่ หากมีข้อมูลที่หายไปจะต้องทำการลบข้อมูลทิ้งไปและผมได้ทำการแยกข้อมูลของแต่ศิลปินออกมาโดยใช้ค่าmeanซึ่งเพลงบางเพลงอาจจะมีการfeat กันก็ได้ทำการแยกชื่อศิลปินออกมาและนำDataของเพลงนั้นไปคิดกับของศิลปินทุกคนที่มีชื่อในเพลงนั้นและผมก็ได้ทำการเอาแนวเพลงที่ศิลปินทำบ่อยสุดจากDatasetโดยการนับและนำเฉพาะแนวเพลงที่มากที่สุดมา")
st.code('''
f = pd.read_csv('data/train.csv')
df = df.dropna()

artist_split = df.assign(
    artists=df['artists'].str.split(';')).explode('artists')

artist_split = artist_split.assign(track_genre=df['track_genre'])

artist_features = artist_split.groupby('artists').agg({
    'popularity': 'mean',
    'duration_ms': 'mean',
    'danceability': 'mean',
    'energy': 'mean',
    'loudness': 'mean',
    'speechiness': 'mean',
    'acousticness': 'mean',
    'instrumentalness': 'mean',
    'liveness': 'mean',
    'valence': 'mean',
    'tempo' : 'mean',
    'key': lambda x: x.mode()[0] if not x.mode().empty else -1, 
    'mode': lambda x: x.mode()[0] if not x.mode().empty else -1, 
    'track_name': 'count',
    'track_genre': lambda x: x.mode()[0]
}).rename(columns={'track_name': 'track_count'})

artist_features['energy_danceability_ratio'] = artist_features['energy'] / (artist_features['danceability'] + 0.01)
artist_features['speech_instrumental_ratio'] = artist_features['speechiness'] / (artist_features['instrumentalness'] + 0.01)
artist_features['acousticness_energy_ratio'] = artist_features['acousticness'] / (artist_features['energy'] + 0.01)
''')

st.divider()
st.write("ต่อมาผมก็ได้ทำการสร้างโมเดล Feedforward Neural Networks (FNN) โดยมีHidden Layers 3 ชั้น")
st.code('''
def build_model(input_shape, num_genres):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_shape,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.LeakyReLU(negative_slope=0.1),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(128, activation='relu'),
        keras.layers.LeakyReLU(negative_slope=0.1),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(64, activation='relu'),
        keras.layers.LeakyReLU(negative_slope=0.1),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(num_genres, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, amsgrad=True),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
''')

st.divider()
st.write("หลักจากสร้างLayerแล้วผมก็ทำการTrain model โดยให้epochs=200คือเทรน200รอบและได้กำหนด early_stopping และ r_schedule และ checkpoint ให้เพื่อป้องกันการเกิดoverfittingและเพื่อเก็บmodelที่ดีที่สุดหลังจากนั้นก็เก็บผลของการเทรนเอาไว้")
st.code('''
MODEL_PATH = "model/genre_classification_model.keras"

if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH)
    trainmodel = None
else:
    model = build_model(input_shape, num_genres)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True
    )

    r_schedule = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10, min_lr=1e-6,
        verbose=1)

    checkpoint = keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )


    trainmodel = model.fit(X_train, y_train, epochs=200, batch_size=64, validation_split=0.2, callbacks=[early_stopping, r_schedule,checkpoint],class_weight = class_weights)

    if trainmodel is not None:
        history_df = pd.DataFrame(trainmodel.history)
        history_df["epoch"] = history_df.index + 1
        history_df.to_csv("model/training_history.csv", index=False)

    model.save(MODEL_PATH)
''')

st.divider()
st.write("หลังจากtrainแล้วผมก็จะทำการload modelมาและนำไปplot graphเพื่อดูว่าmodelที่เราสร้างมีความแม่นยำเท่าไหร่")
st.code('''
    model = keras.models.load_model("model/genre_classification_model.keras", compile=False)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
        plt.figure(figsize=(12, 4))

if trainmodel is not None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    ax[0].plot(trainmodel.history['accuracy'], label='Training Accuracy')
    ax[0].plot(trainmodel.history['val_accuracy'], label='Validation Accuracy')
    ax[0].set_title('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    
    ax[1].plot(trainmodel.history['loss'], label='Training Loss')
    ax[1].plot(trainmodel.history['val_loss'], label='Validation Loss')
    ax[1].set_title('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    
    st.pyplot(fig)
    

elif os.path.exists("model/training_history.csv"):
    history_df = pd.read_csv("model/training_history.csv")
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    ax[0].plot(history_df['accuracy'], label='Training Accuracy')
    ax[0].plot(history_df['val_accuracy'], label='Validation Accuracy')
    ax[0].set_title('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    
    ax[1].plot(history_df['loss'], label='Training Loss')
    ax[1].plot(history_df['val_loss'], label='Validation Loss')
    ax[1].set_title('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    
    st.pyplot(fig)
    
        ''')

st.divider()
st.write("และผมก็จะให้ user สามารถใส่ชื่อศิลปินที่ตัวเองอยากรู้ลงไปได้เพื่อให้ModelทำการPredictว่าศิลปินคนนั้นชอบทำเพลงแนวไหน")
st.code('''
        ef predict_artist_genre(artist_name, df, model, scaler, encoder):

    if artist_name not in df.index:
        return f"ไม่พบข้อมูลของศิลปิน {artist_name}"
    
    try:
        artist_data = df.loc[artist_name]
        
        all_features = []
        for feature in X.columns:
            if feature in artist_data:
                all_features.append(artist_data[feature])
            else:
                all_features.append(0)
                
        features_array = np.array(all_features).reshape(1, -1)
        
        prediction = model.predict(features_array)
        predicted_genre_idx = np.argmax(prediction)
        predicted_genre = label_encoder.classes_[predicted_genre_idx]
        
        confidence = prediction[0][predicted_genre_idx] * 100
        
        top_indices = np.argsort(prediction[0])[-3:][::-1]
        top_genres = [(label_encoder.classes_[idx], prediction[0][idx] * 100) for idx in top_indices]
        
        feature_importances = {}
        for i, feature in enumerate(numerical_features):
            if feature in artist_data and i < len(features_array[0]):
                feature_importances[feature] = features_array[0][i]
        
        strongest_features = sorted(feature_importances.items(),key=lambda x: abs(x[1]),reverse=True)[:5]
        
        return {
            'artist': artist_name,
            'predicted_genre': predicted_genre,
            'confidence': f"{confidence:.2f}%",
            'top_genres': top_genres,
            'notable_characteristics': strongest_features
        }
    except Exception as e:
        return f"ไม่พบข้อมูลของศิลปิน: {str(e)}"

st.subheader("🔍 Predict Artist's Genre")
artist_name = st.text_input(
    "Enter artist name", placeholder="Enter artist name")
if artist_name:
    result = predict_artist_genre(
        artist_name, artist_features, model, scaler, label_encoder)

    if isinstance(result, dict):
        with st.expander("Prediction Results", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Artist:** {result['artist']}")
                st.markdown(f"**Predicted Genre:** {result['predicted_genre']}")
                st.markdown(f"**Confidence:** {result['confidence']}")
                
            with col2:
                st.markdown("**Top 3 Genre Predictions:**")
                for genre, conf in result['top_genres']:
                    st.markdown(f"- {genre}: {conf:.2f}%")
    else:
        st.write(result)
''')

st.link_button("🪴 Neural Network Model", "https://intelligence-project-phoorich.streamlit.app/neural_network_model")
st.link_button("🔗 Source Code", "https://github.com/Soulz157/Intelligence_Project")