import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Pemuatan file .h5
tfidf_filename = "tfidf_vectorizer.h5"
cosine_sim_filename = "cosine_similarity_matrix.h5"
recommender_model_filename = "recommender_model.h5"
history_filename = "training_history.h5"

# Memuat TF-IDF vectorizer
tfidf_vectorizer = joblib.load(tfidf_filename)

# Memuat cosine similarity matrix
cosine_sim_df = joblib.load(cosine_sim_filename)

# Memuat model Collaborative Filtering
model = load_model(recommender_model_filename)

# Memuat riwayat pelatihan
training_history = joblib.load(history_filename)

# Menampilkan informasi dasar
st.title("Dashboard Rekomendasi Makanan")
st.write("Menggunakan TF-IDF, Cosine Similarity, dan Collaborative Filtering untuk memberikan rekomendasi makanan.")

# Menampilkan beberapa data pertama
data = pd.read_csv('/media/primadya/Kerja/dicoding/terapan/submission 2/dataset/food/1662574418893344.csv')
rating = pd.read_csv('/media/primadya/Kerja/dicoding/terapan/submission 2/dataset/food/ratings.csv')

data.columns = data.columns.str.lower()
rating.columns = rating.columns.str.lower()

st.write("### Dataset Makanan")
st.dataframe(data.head(10))

st.write("### Dataset Rating")
st.dataframe(rating.head(10))

# Menampilkan grafik jumlah makanan per tipe
st.write("### Grafik Jumlah Makanan per Tipe")
plt.figure(figsize=(10, 5))
sns.countplot(data=data, x=data['c_type'])
plt.xticks(rotation=90)
st.pyplot(plt)

# Menampilkan grafik jumlah makanan per kategori vegetarian/non-vegetarian
st.write("### Grafik Jumlah Makanan Vegetarian/Non-Vegetarian")
plt.figure(figsize=(5, 3))
sns.countplot(data=data, x=data['veg_non'])
plt.xticks(rotation=90)
st.pyplot(plt)

# Menampilkan grafik jumlah kemunculan rating
st.write("### Grafik Jumlah Rating Makanan")
plt.figure(figsize=(5, 3))
sns.countplot(data=rating, x=rating['rating'])
st.pyplot(plt)

# Menampilkan cosine similarity matrix
st.write("### Cosine Similarity Matrix (Beberapa Contoh)")
st.dataframe(cosine_sim_df.head())

# Menampilkan grafik evaluasi model (RMSE)
st.write("### Grafik Evaluasi Model Collaborative Filtering")
plt.plot(training_history['root_mean_squared_error'], label='Train RMSE')
plt.plot(training_history['val_root_mean_squared_error'], label='Validation RMSE')
plt.title('Model Evaluation - RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
st.pyplot(plt)

# Fungsi untuk memberikan rekomendasi makanan
def food_recommendations(input_value, k=4):
    input_value = input_value.strip().lower()

    # Jika input_value ada dalam similarity_data
    matching_items = [name for name in cosine_sim_df.columns if input_value in name.lower()]

    if matching_items:
        closest_df = pd.DataFrame({'name': matching_items})
        return closest_df.head(k)
    else:
        return f"Tidak ada rekomendasi untuk '{input_value}'."

# Form input untuk pencarian rekomendasi makanan
input_value = st.text_input("Masukkan nama makanan atau jenis makanan yang Anda suka:", "")
if input_value:
    recommendations = food_recommendations(input_value)
    st.write(f"### Rekomendasi untuk '{input_value}':")
    if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
        st.dataframe(recommendations)
    else:
        st.write(recommendations)

# Menambahkan interaksi untuk prediksi berdasarkan model Collaborative Filtering
if st.button('Dapatkan Rekomendasi Makanan dengan Collaborative Filtering'):
    user_id = rating['user_id'].sample(1).iloc[0]  # Ambil sample user ID secara acak
    food_visited_by_user = rating[rating['user_id'] == user_id]
    
    st.write(f"### Rekomendasi untuk pengguna {user_id}:")
    # Prediksi makanan yang belum dikunjungi pengguna
    food_not_visited = data[~data['food_id'].isin(food_visited_by_user.food_id.values)]['food_id']
    food_not_visited = list(set(food_not_visited))
    
    user_encoder = user_id  # Mapping user ID ke encoder
    food_not_visited_data = np.array([[user_encoder, food] for food in food_not_visited])
    
    predictions = model.predict(food_not_visited_data)
    
    # Tampilkan rekomendasi berdasarkan rating tertinggi
    top_food_ids = np.argsort(predictions.flatten())[-10:][::-1]
    recommended_food_ids = [food_not_visited[i] for i in top_food_ids]
    
    for food_id in recommended_food_ids:
        food_name = data[data['food_id'] == food_id]['name'].values[0]
        st.write(f"Food: {food_name}")
