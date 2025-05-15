import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import joblib

# ----------------------------
# Kelas Custom untuk Model CF
# ----------------------------
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_food, embedding_size=50, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_food = num_food
        self.embedding_size = embedding_size

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
        dot_product = tf.reduce_sum(user_vector * food_vector, axis=1, keepdims=True)
        x = dot_product + user_bias + food_bias
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_users": self.num_users,
            "num_food": self.num_food,
            "embedding_size": self.embedding_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            num_users=config["num_users"],
            num_food=config["num_food"],
            embedding_size=config.get("embedding_size", 50)
        )


# ----------------------------
# Load Model dan Objek
# ----------------------------

try:
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    cosine_sim_df = joblib.load("cosine_similarity_matrix.pkl")
    training_history = joblib.load("training_history.pkl")
except Exception as e:
    st.error(f"❌ Terjadi kesalahan saat memuat file: {e}")

# Load model CF dengan custom object
try:
    get_custom_objects().update({'RecommenderNet': RecommenderNet})
    with tf.keras.utils.custom_object_scope({'RecommenderNet': RecommenderNet}):
        model = load_model("recommendernew.h5")
except Exception as e:
    st.error(f"❌ Gagal memuat model Collaborative Filtering: {e}")
    model = None

# ----------------------------
# Load Data
# ----------------------------

@st.cache_data
def load_data():
    data = pd.read_csv("dataset/food.csv")
    rating = pd.read_csv("dataset/ratings.csv")
    data.columns = data.columns.str.lower()
    rating.columns = rating.columns.str.lower()
    return data, rating

data, rating = load_data()

# ----------------------------
# Tampilan Streamlit
# ----------------------------
st.title("🍽️ Dashboard Rekomendasi Makanan")
st.write("Menggunakan **TF-IDF**, **Cosine Similarity**, dan **Collaborative Filtering** untuk memberikan rekomendasi makanan yang sesuai dengan selera Anda.")

st.subheader("📊 Dataset Makanan")
st.dataframe(data.head(10))

st.subheader("⭐ Dataset Rating")
st.dataframe(rating.head(10))

# ----------------------------
# Fungsi Content-Based Filtering
# ----------------------------
def food_recommendations(input_value, k=4):
    input_value = input_value.strip().lower()
    matching_items = [name for name in cosine_sim_df.columns if input_value in name.lower()]
    if matching_items:
        return pd.DataFrame({'name': matching_items[:k]})
    else:
        return None

input_value = st.text_input("🔍 Masukkan nama makanan yang Anda suka:")
if input_value:
    recommendations = food_recommendations(input_value)
    st.subheader(f"🍱 Rekomendasi untuk '{input_value}':")
    if recommendations is not None and not recommendations.empty:
        st.dataframe(recommendations)
    else:
        st.warning(f"Tidak ditemukan rekomendasi untuk '{input_value}'.")

# ----------------------------
# Collaborative Filtering
# ----------------------------
if st.button('🎯 Dapatkan Rekomendasi dengan Collaborative Filtering'):
    if model is not None:
        with st.spinner("🔄 Menghitung rekomendasi..."):
            user_id = rating['user_id'].sample(1).iloc[0]
            food_visited_by_user = rating[rating['user_id'] == user_id]
            food_not_visited = data[~data['food_id'].isin(food_visited_by_user['food_id'].values)]['food_id'].unique()

            user_encoder = user_id  # Sesuaikan jika pakai LabelEncoder
            food_not_visited_data = np.array([[user_encoder, food_id] for food_id in food_not_visited])
            predictions = model.predict(food_not_visited_data, verbose=0)

            top_indices = np.argsort(predictions.flatten())[-10:][::-1]
            recommended_ids = [food_not_visited[i] for i in top_indices]

            st.subheader(f"📌 Rekomendasi untuk Pengguna {user_id}:")
            for fid in recommended_ids:
                row = data[data['food_id'] == fid]
                if not row.empty:
                    name = row['name'].values[0]
                    ftype = row['c_type'].values[0]
                    st.markdown(f"- **{name}** ({ftype})")
    else:
        st.error("Model Collaborative Filtering belum berhasil dimuat.")
