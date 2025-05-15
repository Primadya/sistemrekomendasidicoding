import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load Model dan Objek TF-IDF
# ----------------------------
try:
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    cosine_sim_df = joblib.load("cosine_similarity_matrix.pkl")
    training_history = joblib.load("training_history.pkl")
except Exception as e:
    st.error(f"❌ Terjadi kesalahan saat memuat file: {e}")

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("dataset/food/1662574418893344.csv")
    rating = pd.read_csv("dataset/food/ratings.csv")
    data.columns = data.columns.str.lower()
    rating.columns = rating.columns.str.lower()
    return data, rating

data, rating = load_data()

# ----------------------------
# Tampilan Streamlit
# ----------------------------
st.title("🍽️ Dashboard Rekomendasi Makanan")
st.write("Menggunakan **TF-IDF** dan **Cosine Similarity** untuk memberikan rekomendasi makanan yang sesuai dengan selera Anda.")

st.subheader("📊 Dataset Makanan")
st.dataframe(data)  # Menampilkan semua data makanan

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
