"""Pipeline analisis sentimen: dari berkas scraping sampai prediksi NB dan RF.
Logikanya sama dengan skrip penelitian (02_bersihkan, t1-t7, 10_label_kamus, 11_split, 12_gold_set,
14_tfidf, 15_latih_model); di sini dibungkus menjadi fungsi agar bisa dipanggil dari aplikasi."""
import io
import json
import re

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import kamus_data as K

SEED = 42
KOLOM_WAJIB = ["content", "score", "at", "reviewCreatedVersion"]
LABEL_VALID = ["positif", "netral", "negatif"]


# ============================ 1. Baca dan validasi berkas ============================
def baca_berkas(file, nama):
    """Membaca .xlsx/.csv hasil scraping dan memastikan kolom wajib ada."""
    nama = nama.lower()
    try:
        if nama.endswith(".csv"):
            df = pd.read_csv(file)
        elif nama.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            raise ValueError("Format berkas harus .xlsx atau .csv.")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Berkas tidak dapat dibaca: {e}")
    hilang = [k for k in KOLOM_WAJIB if k not in df.columns]
    if hilang:
        raise ValueError("Kolom wajib tidak ditemukan: " + ", ".join(hilang) +
                         ". Kolom yang diperlukan: " + ", ".join(KOLOM_WAJIB) + ".")
    if df.empty:
        raise ValueError("Berkas tidak berisi data.")
    return df


# ============================ 2. Pembersihan data ============================
def bersihkan(df):
    """Isi versi kosong, buang baris tidak valid dan duplikat, beri id tetap.
    Duplikat = content + tanggal (per hari) + score + versi sama persis; yang pertama disimpan."""
    ringkas = {"awal": len(df)}
    d = df.copy()
    d["reviewCreatedVersion"] = d["reviewCreatedVersion"].fillna("Tidak Diketahui").astype(str)
    d["at"] = pd.to_datetime(d["at"], errors="coerce")
    d["score"] = pd.to_numeric(d["score"], errors="coerce")
    valid = d["content"].notna() & d["at"].notna() & d["score"].between(1, 5)
    ringkas["tidak_valid"] = int((~valid).sum())
    d = d[valid].copy()
    d["score"] = d["score"].astype(int)
    d["tanggal"] = d["at"].dt.date
    sebelum = len(d)
    d = d.drop_duplicates(subset=["content", "tanggal", "score", "reviewCreatedVersion"], keep="first")
    ringkas["duplikat"] = sebelum - len(d)
    d = d.reset_index(drop=True)
    d.insert(0, "id", d.index + 1)
    return d[["id", "content", "score", "at", "tanggal", "reviewCreatedVersion"]], ringkas


# ============================ 3. Pra-pemrosesan teks ============================
def _cleaning(t):
    t = re.sub(r"https?://\S+|www\.\S+", " ", str(t))   # hapus URL
    t = re.sub(r"[^a-z\s]", " ", t)                      # hapus tanda baca, angka, simbol, emoji
    t = re.sub(r"([a-z])\1{2,}", r"\1", t)               # huruf berulang 3x atau lebih jadi 1
    return re.sub(r"\s+", " ", t).strip()


def _norm_token(w):
    if w in K.KAMUS:
        return K.KAMUS[w]
    if w.endswith("nya") and len(w) > 4:
        b = w[:-3]
        if b in K.KAMUS:
            return K.KAMUS[b]
        if b in K.DOMAIN:
            return b
    return w


def _normalisasi(t):
    return " ".join(_norm_token(w) for w in t.split())


def praproses(df, peta_stem, progres=None):
    """Casefolding, cleaning, normalisasi, tokenizing, stopword, stemming, lalu buang ulasan kosong.
    peta_stem: dict kata->stem (kamus stem). Kata yang belum ada di-stem dengan Sastrawi dan DITAMBAHKAN
    ke peta_stem. Mengembalikan (df, kata_baru) dengan kata_baru = dict kata->stem yang baru dibuat."""
    stop = (set(StopWordRemoverFactory().get_stop_words()) | K.EXTRA) - K.NEGASI - {"bisa"}
    token = []
    for t in df["content"].astype(str):
        t = _normalisasi(_cleaning(t.lower()))
        token.append([w for w in t.split() if w not in stop and len(w) > 1])

    vocab = sorted({w for toks in token for w in toks})
    belum = [w for w in vocab if w not in peta_stem]
    kata_baru = {}
    if belum:
        stemmer = StemmerFactory().create_stemmer()
        for i, w in enumerate(belum, 1):
            kata_baru[w] = peta_stem[w] = stemmer.stem(w)
            if progres and (i % 25 == 0 or i == len(belum)):
                progres(i, len(belum))

    stem = [["perbaiki" if a in K.PERBAIKI else peta_stem[a] for a in toks] for toks in token]
    d = df.copy()
    d["teks_prep"] = [" ".join(s) for s in stem]
    d["_asal"] = token
    d["_stem"] = stem
    d = d[d["_stem"].apply(len) > 0].reset_index(drop=True)   # buang ulasan kosong; id tidak diurutkan ulang
    return d, kata_baru


# ============================ 4. Pelabelan berbasis kamus (InSet) ============================
def _baca_inset(path):
    d = pd.read_csv(path, sep="\t", encoding="utf-8")
    d.columns = ["word", "weight"]
    d["word"] = (d["word"].astype(str).str.lower().str.replace(r"\(.*?\)", "", regex=True).str.strip())
    d = d[~d["word"].str.contains(" ")]                    # entri lebih dari satu kata dilewati
    return d.drop_duplicates("word").set_index("word")["weight"].to_dict()


def bangun_inset(path_pos, path_neg):
    pos_w, neg_w = _baca_inset(path_pos), _baca_inset(path_neg)
    ambigu = set(pos_w) & set(neg_w)                       # ada di kedua file -> dibuang dari skor
    inset = {w: v for w, v in {**pos_w, **neg_w}.items() if w not in ambigu}
    inset.update(K.TAMBAHAN)
    for w in K.NETRAL | K.NEGATOR:
        inset.pop(w, None)
    return inset


def beri_label(df, inset):
    """Skor kamus dengan pembalikan oleh negator (jendela 2 token) lalu label positif/negatif/netral."""
    def bobot(x, y):
        if x in inset:
            return inset[x]
        if y in inset:
            return inset[y]
        return 0

    def skor(asal, stem):
        total, sisa = 0, 0
        for x, y in zip(asal, stem):
            if x in K.NEGATOR:
                sisa = K.JENDELA
                continue
            w = bobot(x, y)
            if sisa > 0:
                if w != 0:
                    w = -w
                    sisa = 0
                else:
                    sisa -= 1
            total += w
        return total

    d = df.copy()
    d["skor"] = [skor(a, s) for a, s in zip(d["_asal"], d["_stem"])]
    d["label_kamus"] = d["skor"].apply(lambda s: "positif" if s > 0 else ("negatif" if s < 0 else "netral"))
    return d.drop(columns=["_asal", "_stem"])


# ============================ 5. Pembagian data dan gold set ============================
def bagi_data(df):
    """Stratified 80:20 menurut label kamus (seed 42)."""
    _, uji = train_test_split(df, test_size=0.2, stratify=df["label_kamus"], random_state=SEED)
    d = df.copy()
    d["partisi"] = "latih"
    d.loc[d["id"].isin(uji["id"]), "partisi"] = "uji"
    return d


def batas_gold(df):
    """Rentang ukuran gold set yang diizinkan: setiap kelas butuh minimal 1 ulasan di data uji."""
    uji = df[df["partisi"] == "uji"]
    return 100, min(1500, len(uji) - len(uji["label_kamus"].unique()) - 1)


def ambil_gold(df, n):
    """Gold set: n ulasan dari DATA UJI, stratified menurut label kamus (seed 42). Mengembalikan daftar id."""
    uji = df[df["partisi"] == "uji"]
    _, gold = train_test_split(uji, test_size=int(n), stratify=uji["label_kamus"], random_state=SEED)
    return sorted(gold["id"].tolist())


def lembar_gold(df, ids):
    """Berkas Excel untuk pelabelan manual: tanpa label kamus dan tanpa rating, urutan diacak."""
    from openpyxl.styles import Alignment
    from openpyxl.worksheet.datavalidation import DataValidation
    gold = df[df["id"].isin(ids)][["id", "content"]]
    lembar = gold.sample(frac=1, random_state=SEED).reset_index(drop=True)
    lembar["label_manual"] = ""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        lembar.to_excel(w, index=False, sheet_name="label")
        ws = w.sheets["label"]
        ws.column_dimensions["A"].width = 8
        ws.column_dimensions["B"].width = 110
        ws.column_dimensions["C"].width = 16
        for row in ws.iter_rows(min_row=2, max_col=2):
            row[1].alignment = Alignment(wrap_text=True, vertical="top")
        dv = DataValidation(type="list", formula1='"positif,netral,negatif"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"C2:C{len(lembar) + 1}")
    return buf.getvalue()


def baca_gold(file, id_diharapkan):
    """Membaca gold set yang sudah dilabeli. Memeriksa id, label kosong, dan label tidak valid.
    Mengembalikan DataFrame (id, label_manual); bila ada masalah, ValueError dengan pesan yang jelas."""
    try:
        d = pd.read_excel(file)
    except Exception as e:
        raise ValueError(f"Berkas tidak dapat dibaca: {e}")
    for k in ("id", "label_manual"):
        if k not in d.columns:
            raise ValueError(f"Kolom '{k}' tidak ditemukan. Gunakan lembar yang diunduh dari halaman ini.")
    d = d[["id", "label_manual"]].copy()
    d["label_manual"] = d["label_manual"].astype("string").str.strip().str.lower()
    ada, harus = set(d["id"].dropna().astype(int)), set(id_diharapkan)
    if d["id"].duplicated().any():
        raise ValueError("Ada id yang muncul lebih dari sekali.")
    if ada != harus:
        raise ValueError(f"Daftar id tidak sama dengan gold set yang diminta: kurang {len(harus - ada)} id, "
                         f"kelebihan {len(ada - harus)} id. Gunakan lembar yang diunduh dari halaman ini "
                         "dan jangan menambah atau menghapus baris.")
    kosong = d[d["label_manual"].isna() | (d["label_manual"] == "")]
    if len(kosong):
        contoh = ", ".join(str(int(i)) for i in kosong["id"].head(5))
        raise ValueError(f"{len(kosong)} baris belum diberi label (contoh id: {contoh}).")
    salah = d[~d["label_manual"].isin(LABEL_VALID)]
    if len(salah):
        contoh = ", ".join(f"{int(i)}='{v}'" for i, v in zip(salah["id"].head(5), salah["label_manual"].head(5)))
        raise ValueError(f"{len(salah)} baris berisi label yang tidak dikenal (contoh: {contoh}). "
                         "Label yang valid: positif, netral, negatif.")
    d["id"] = d["id"].astype(int)
    d["label_manual"] = d["label_manual"].astype(str)
    return d.sort_values("id").reset_index(drop=True)


# ============================ 6. TF-IDF, pelatihan, prediksi ============================
def latih_prediksi(df, progres=None):
    """TF-IDF (fit hanya pada data latih), latih NB dan RF pada label kamus, prediksi SEMUA ulasan."""
    latih = (df["partisi"] == "latih").to_numpy()
    y = df["label_kamus"].to_numpy()
    if progres:
        progres("TF-IDF")
    vec = TfidfVectorizer(min_df=2)
    vec.fit(df.loc[latih, "teks_prep"])
    X = vec.transform(df["teks_prep"])
    if progres:
        progres("Naive Bayes")
    nb = MultinomialNB().fit(X[latih], y[latih])
    if progres:
        progres("Random Forest")
    rf = RandomForestClassifier(random_state=SEED, n_jobs=-1).fit(X[latih], y[latih])
    if progres:
        progres("Prediksi")
    d = df.copy()
    d["pred_nb"] = nb.predict(X)
    d["pred_rf"] = rf.predict(X)
    return d, {"fitur": int(X.shape[1])}
