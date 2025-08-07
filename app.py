import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# モデルをロード（事前訓練されたモデル）
model = tf.keras.applications.MobileNetV2(weights='imagenet')

def preprocess_image(img):
    img = img.resize((224, 224))  # モデルの入力サイズにリサイズ
    img_array = np.array(img)  # NumPy 配列に変換
    img_array = np.expand_dims(img_array, axis=0)  # バッチ次元を追加
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)  # モデルに適した前処理
    return img_array

def predict_image(img):
    preprocessed_img = preprocess_image(img)
    preds = model.predict(preprocessed_img)  # 予測を実行
    decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds)[0]
    return decoded_preds

st.title("InstaRecipe")

uploaded_image = st.file_uploader("画像をアップロード", type=["jpg", "png"])

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption="アップロードした画像", use_container_width=True)
    
    # 画像認識を行う
    predictions = predict_image(img)
    
    # 結果を表示
    st.write("検出された物体:")
    for pred in predictions:
        st.write(f"{pred[1]} (確率: {pred[2]*100:.2f}%)")
