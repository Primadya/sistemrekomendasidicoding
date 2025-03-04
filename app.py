import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import joblib

# Misalkan RecommenderNet adalah kelas atau layer kustom
# Anda harus mendefinisikan atau mengimpor RecommenderNet jika itu adalah layer kustom
# Contoh definisi RecommenderNet (jika itu model kustom)

class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_food, embedding_size=50, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.user_embedding = tf.keras.layers.Embedding(
            num_users, embedding_size, embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )
        self.user_bias = tf.keras.layers.Embedding(num_users, 1)
        self.food_embedding = tf.keras.layers.Embedding(
            num_food, embedding_size, embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )
        self.food_bias = tf.keras.layers.Embedding(num_food, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        food_vector = self.food_embedding(inputs[:, 1])
        food_bias = self.food_bias(inputs[:, 1])
        dot_product = tf.tensordot(user_vector, food_vector, axes=2)
        x = dot_product + user_bias + food_bias
        return x  # Output tanpa sigmoid untuk regresi

# Pemuatan file .h5
tfidf_filename = "tfidf_vectorizer.h5"
cosine_sim_filename = "cosine_similarity_matrix.h5"
recommender_model_filename = "recommender_model.h5"
history_filename = "training_history.h5"

# Memuat TF-IDF vectorizer
tfidf_vectorizer = joblib.load(tfidf_filename)

# Memuat cosine similarity matrix
cosine_sim_df = joblib.load(cosine_sim_filename)

# Memuat objek kustom di model
get_custom_objects().update({'RecommenderNet': RecommenderNet})

# Memuat model Collaborative Filtering dengan custom object scope
with tf.keras.utils.custom_object_scope({'RecommenderNet': RecommenderNet}):
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

# Fungsi untuk memberikan rekomendasi makanan berdasarkan cosine similarity
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
        food_type = data[data['food_id'] == food_id]['c_type'].values[0]  # Menambahkan informasi jenis makanan
        st.write(f"Food: {food_name}, Type: {food_type}")
