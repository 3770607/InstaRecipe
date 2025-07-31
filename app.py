import streamlit as st

st.title("InstaRecipe")

uploaded_image = st.file_uploader("画像をアップロード", type=["jpg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="アップロードした画像", use_column_width=True)
    
    # 画像認識結果の表示
    # モデルからの予測結果（仮）
    st.write("検出された食材: 例: トマト, じゃがいも, にんじん")
