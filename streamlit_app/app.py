from pathlib import Path
import re

import joblib
import numpy as np
import streamlit as st
import trafilatura
from gensim.models import Word2Vec
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


# ==========================================================
# KONFIGURASI
# ==========================================================

st.set_page_config(
    page_title="Klasifikasi Berita Detik.com",
    page_icon="📰",
    layout="wide"
)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "modul"

W2V_PATH = MODEL_DIR / "w2v_skenario1_final.model"
NB_PATH = MODEL_DIR / "naive_bayes_skenario1.pkl"


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_models():
    model_w2v = Word2Vec.load(str(W2V_PATH))
    model_nb = joblib.load(str(NB_PATH))
    return model_w2v, model_nb


# ==========================================================
# PREPROCESSING
# Skenario 1:
# hapus angka = True
# hapus stopword = True
# stemming = True
# ==========================================================

stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

def preprocessing_text(text):
    text = text.lower()

    # Menghapus URL
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Menghapus angka
    text = re.sub(r"\d+", " ", text)

    # Menyisakan huruf dan spasi
    text = re.sub(r"[^a-z\s]", " ", text)

    # Menghapus spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    # Stopword removal
    text = stopword_remover.remove(text)

    # Tokenisasi sederhana
    tokens = text.split()

    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]

    return tokens


# ==========================================================
# MEAN POOLING WORD2VEC
# ==========================================================

def document_vector(tokens, model_w2v):
    vectors = []

    for word in tokens:
        if word in model_w2v.wv:
            vectors.append(model_w2v.wv[word])

    if not vectors:
        return np.zeros(model_w2v.vector_size)

    return np.mean(vectors, axis=0)


# ==========================================================
# LABEL PREDIKSI
# ==========================================================

def format_prediction(prediction, classes):
    value = prediction[0]

    if isinstance(value, str):
        value_lower = value.lower()

        if "sport" in value_lower:
            return "SPORT"
        if "finance" in value_lower or "finans" in value_lower:
            return "FINANCE"

        return str(value).upper()

    if len(classes) == 2:
        try:
            numeric_value = int(value)

            if numeric_value == 0:
                return "SPORT"
            if numeric_value == 1:
                return "FINANCE"
        except (ValueError, TypeError):
            pass

    return str(value)


# ==========================================================
# SCRAPING ARTIKEL
# ==========================================================

def extract_article(url):
    downloaded = trafilatura.fetch_url(url)

    if not downloaded:
        return None

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        favor_precision=True
    )

    return text


# ==========================================================
# HEADER
# ==========================================================

st.title("📰 Klasifikasi Berita Detik.com")

st.markdown(
    """
    Aplikasi klasifikasi berita menggunakan **Word2Vec Skip-gram**
    dan **Naive Bayes** berdasarkan model hasil Tugas 3.
    """
)

st.divider()


# ==========================================================
# INPUT URL
# ==========================================================

st.subheader("🔗 Masukkan URL Berita")

url = st.text_input(
    "URL artikel Detik.com",
    placeholder="https://sport.detik.com/... atau https://finance.detik.com/..."
)

classify_button = st.button(
    "🔍 Klasifikasikan Berita",
    type="primary",
    use_container_width=True
)


# ==========================================================
# PROSES KLASIFIKASI
# ==========================================================

if classify_button:

    if not url.strip():
        st.warning("Silakan masukkan URL artikel terlebih dahulu.")
        st.stop()

    if "detik.com" not in url.lower():
        st.warning("Gunakan URL artikel dari Detik.com.")
        st.stop()

    try:

        # --------------------------------------------------
        # LOAD MODEL
        # --------------------------------------------------

        with st.spinner("Memuat model Word2Vec dan Naive Bayes..."):
            model_w2v, model_nb = load_models()

        st.success("Model berhasil dimuat.")

        # --------------------------------------------------
        # SCRAPING
        # --------------------------------------------------

        with st.spinner("Mengunduh halaman web..."):
            article_text = extract_article(url)

        if not article_text:
            st.error(
                "Artikel tidak dapat diekstrak. "
                "Pastikan URL merupakan halaman artikel Detik.com yang dapat diakses."
            )
            st.stop()

        # --------------------------------------------------
        # PREPROCESSING
        # --------------------------------------------------

        with st.spinner("Membersihkan dan melakukan preprocessing teks..."):
            tokens = preprocessing_text(article_text)

        if not tokens:
            st.error("Tidak ditemukan teks yang dapat diproses.")
            st.stop()

        # --------------------------------------------------
        # WORD2VEC
        # --------------------------------------------------

        with st.spinner("Mengekstrak fitur menggunakan Word2Vec Skip-gram..."):
            doc_vector = document_vector(tokens, model_w2v)

        # --------------------------------------------------
        # NAIVE BAYES
        # --------------------------------------------------

        with st.spinner("Mengklasifikasikan dengan Naive Bayes..."):
            prediction = model_nb.predict([doc_vector])

        result = format_prediction(
            prediction,
            model_nb.classes_
        )

        # --------------------------------------------------
        # HASIL
        # --------------------------------------------------

        st.divider()

        st.subheader("📊 Hasil Klasifikasi")

        if result == "SPORT":
            st.success("Kategori Berita: **SPORT**")
        elif result == "FINANCE":
            st.success("Kategori Berita: **FINANCE**")
        else:
            st.info(f"Kategori Berita: **{result}**")

        # --------------------------------------------------
        # DETAIL PROSES
        # --------------------------------------------------

        with st.expander("📄 Lihat teks artikel"):
            st.write(article_text)

        with st.expander("🧹 Lihat hasil preprocessing"):
            st.write(" ".join(tokens))

        with st.expander("🔢 Informasi fitur"):
            st.write(f"Jumlah token setelah preprocessing: **{len(tokens)}**")
            st.write(f"Dimensi vektor dokumen: **{len(doc_vector)}**")
            st.write(f"Word2Vec vector size: **{model_w2v.vector_size}**")

    except FileNotFoundError as e:
        st.error(
            f"File model tidak ditemukan: {e}. "
            "Pastikan model berada di folder modul."
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan saat menjalankan klasifikasi: {e}")


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Tugas 3 Pencarian dan Penambangan Web | "
    "Word2Vec Skip-gram + Naive Bayes"
)
