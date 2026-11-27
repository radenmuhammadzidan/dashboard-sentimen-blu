import os
import re
import time

import numpy as np
import pandas as pd
import streamlit as st
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Uji Teknis", layout="centered")
st.title("Uji teknis: batas kemampuan server")
st.caption("Halaman sementara untuk mengukur waktu dan memori. Label dibuat acak "
           "hanya untuk mengukur waktu, BUKAN hasil penelitian.")


def memori_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            return float("nan")


berkas = st.file_uploader("Unggah file ulasan (xlsx atau csv, harus ada kolom 'content')",
                          type=["xlsx", "csv"])
n = st.number_input("Jumlah ulasan yang diuji", min_value=500, max_value=20000,
                    value=2000, step=500)
st.caption("Mulai dari 2000, lalu naikkan bertahap sampai 20000 untuk melihat batasnya.")

if st.button("Mulai uji", disabled=berkas is None):
    hasil = []
    bar = st.progress(0.0, text="Mulai...")

    def catat(nama, t0, langkah):
        hasil.append({"Tahap": nama,
                      "Waktu (detik)": round(time.time() - t0, 1),
                      "Memori proses (MB)": round(memori_mb())})
        bar.progress(langkah / 8, text=f"Selesai: {nama}")

    # 1. baca berkas
    t0 = time.time()
    if berkas.name.lower().endswith(".csv"):
        df = pd.read_csv(berkas)
    else:
        df = pd.read_excel(berkas)
    if "content" not in df.columns:
        st.error("Kolom 'content' tidak ditemukan.")
        st.stop()
    teks = df["content"].astype(str).head(int(n)).tolist()
    catat("1. Baca berkas", t0, 1)

    # 2. cleaning
    t0 = time.time()
    bersih = [re.sub(r"\s+", " ", re.sub(r"([a-z])\1{2,}", r"\1",
              re.sub(r"[^a-z\s]", " ", re.sub(r"https?://\S+|www\.\S+", " ", t.lower())))).strip()
              for t in teks]
    catat("2. Cleaning", t0, 2)

    # 3. tokenizing + stopword
    t0 = time.time()
    stop = set(StopWordRemoverFactory().get_stop_words()) - {"tidak", "tak", "bukan", "belum", "tanpa", "bisa"}
    token = [[w for w in t.split() if w not in stop and len(w) > 1] for t in bersih]
    catat("3. Tokenizing dan stopword", t0, 3)

    # 4. stemming (sekali per kata unik)
    t0 = time.time()
    kosakata = sorted({w for toks in token for w in toks})
    stemmer = StemmerFactory().create_stemmer()
    peta = {w: stemmer.stem(w) for w in kosakata}
    stem = [" ".join(peta[w] for w in toks) for toks in token]
    catat(f"4. Stemming ({len(kosakata)} kata unik)", t0, 4)

    # 5. TF-IDF
    t0 = time.time()
    vec = TfidfVectorizer(min_df=2)
    X = vec.fit_transform(stem)
    catat(f"5. TF-IDF ({X.shape[1]} fitur)", t0, 5)

    y = np.random.RandomState(42).choice(["positif", "negatif", "netral"], size=X.shape[0])

    # 6. Naive Bayes
    t0 = time.time()
    nb = MultinomialNB().fit(X, y)
    catat("6. Latih Naive Bayes", t0, 6)

    # 7. Random Forest
    t0 = time.time()
    rf = RandomForestClassifier(random_state=42, n_jobs=-1).fit(X, y)
    catat("7. Latih Random Forest", t0, 7)

    # 8. prediksi
    t0 = time.time()
    nb.predict(X)
    rf.predict(X)
    catat("8. Prediksi NB dan RF", t0, 8)

    st.success(f"Selesai untuk {len(teks)} ulasan.")
    st.dataframe(pd.DataFrame(hasil), hide_index=True, use_container_width=True)
    st.metric("Total waktu (detik)", round(sum(h["Waktu (detik)"] for h in hasil), 1))
