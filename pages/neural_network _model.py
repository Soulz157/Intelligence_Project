import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


df = pd.read_csv('data/train.csv')
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

artist_features = artist_features[artist_features['track_count'] >= 3]

numerical_features = ['popularity', 'duration_ms', 'danceability', 'energy', 'loudness',
                      'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 
                      'tempo', 'track_count', 'energy_danceability_ratio', 
                      'speech_instrumental_ratio', 'acousticness_energy_ratio']


scaler = StandardScaler()
artist_features[numerical_features] = scaler.fit_transform(
    artist_features[numerical_features])


categorical_features = ['key', 'mode']
for cat_feature in categorical_features:
    dummies = pd.get_dummies(artist_features[cat_feature], prefix=cat_feature)
    artist_features = pd.concat([artist_features, dummies], axis=1)

X = artist_features.drop(columns=['track_genre'] + categorical_features)
y = artist_features['track_genre']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_genres = len(label_encoder.classes_)

y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=num_genres)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded)

class_counts = np.sum(y_train, axis=0)
class_weights = {i: (1 / class_counts[i]) * (len(y_train) / num_genres) for i in range(num_genres)}


input_shape = X_train.shape[1]
num_genres = y_train.shape[1]


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


    trainmodel = model.fit(X_train, y_train, epochs=200, batch_size=64,
                           validation_split=0.2, callbacks=[early_stopping, r_schedule,checkpoint],class_weight = class_weights)

    if trainmodel is not None:
        history_df = pd.DataFrame(trainmodel.history)
        history_df["epoch"] = history_df.index + 1
        history_df.to_csv("model/training_history.csv", index=False)

    model.save(MODEL_PATH)

test_loss, test_accuracy = model.evaluate(X_test, y_test)


def predict_artist_genre(artist_name, df, model, scaler, encoder):

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
        
        strongest_features = sorted(feature_importances.items(), 
                                   key=lambda x: abs(x[1]), 
                                   reverse=True)[:5]
        
        return {
            'artist': artist_name,
            'predicted_genre': predicted_genre,
            'confidence': f"{confidence:.2f}%",
            'top_genres': top_genres,
            'notable_characteristics': strongest_features
        }
    except Exception as e:
        return f"ไม่พบข้อมูลของศิลปิน: {str(e)}"
    
    
model = keras.models.load_model(
    "model/genre_classification_model.keras", compile=False)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

st.title('🎵 Music Genre Classification')
st.write('อันนี้จะเป็นโมเดลที่ผมได้ทำไว้เพื่อทำนายแนวเพลงที่ศิลปินคนนั้นๆชอบทำโดยสามารถใส่ชื่อศิลปินเข้าไปได้เลยหรือสามารถเข้าไปcopyชื่อของศิลปินได้โดยการกดที่ Dataset Information')

with st.expander("Dataset Information"):
    st.write(f"Total artists: {len(artist_features)}")
    st.write(f"Total genres: {num_genres}")
    st.write(f"Features used: {len(X.columns)}")

    all_artists_df = pd.DataFrame(artist_features.index.tolist(), columns=["Artist Name"])
    st.write("All Artists:")
    st.dataframe(all_artists_df, height=300)


if st.checkbox("Show Sample Data"):
        st.dataframe(artist_features.head())



st.subheader("📊 Model Performance")
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

