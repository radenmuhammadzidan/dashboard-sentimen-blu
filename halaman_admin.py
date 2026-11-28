from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

import auth
import dw
import pipeline as P
from db import get_engine

auth.wajib_admin()

BASE = Path(__file__).parent
engine = get_engine()


def angka(n):
    return f"{int(n):,}".replace(",", ".")


def margin_galat(n, populasi):
    """Perkiraan margin galat 95% untuk akurasi (p = 0,5, dengan koreksi populasi terbatas), dalam poin persen."""
    return 1.96 * (0.25 / n) ** 0.5 * ((populasi - n) / (populasi - 1)) ** 0.5 * 100


def wib(waktu):
    return (waktu + timedelta(hours=7)).strftime("%d/%m/%Y %H:%M WIB")


def muat_data_sementara(info):
    """Data sementara dimuat sekali per sesi (diganti bila ada unggahan baru)."""
    kunci = ("stg", info["waktu"])
    if st.session_state.get("stg_kunci") != kunci:
        st.session_state["stg_df"] = dw.muat_staging(engine)
        st.session_state["stg_kunci"] = kunci
    return st.session_state["stg_df"]


def proses_berkas(berkas):
    """Berkas scraping -> bersih -> pra-proses -> label kamus -> pembagian data -> simpan sebagai data sementara."""
    with st.status("Memproses berkas...", expanded=True) as status:
        try:
            st.write("1/5 Membaca dan memeriksa berkas")
            mentah = P.baca_berkas(berkas, berkas.name)
            bersih, ringkas = P.bersihkan(mentah)
            if len(bersih) < 600:
                raise ValueError(f"Data terlalu sedikit: hanya {len(bersih)} ulasan valid setelah dibersihkan "
                                 "(minimal 600).")

            st.write("2/5 Pra-pemrosesan teks (stemming kata yang belum dikenal membutuhkan waktu)")
            bar = st.progress(0.0, text="Memuat kamus stem")
            peta = dw.muat_peta_stem(engine)

            def kemajuan(i, total):
                bar.progress(i / total, text=f"Stemming kata baru: {angka(i)} dari {angka(total)}")

            data, kata_baru = P.praproses(bersih, peta, kemajuan)
            bar.progress(1.0, text=f"Pra-pemrosesan selesai ({angka(len(kata_baru))} kata baru)")
            dw.simpan_kata_baru(engine, kata_baru)

            st.write("3/5 Pelabelan berbasis kamus InSet")
            inset = P.bangun_inset(BASE / "kamus" / "positive.tsv", BASE / "kamus" / "negative.tsv")
            data = P.beri_label(data, inset)
            sebaran = data["label_kamus"].value_counts()
            if len(sebaran) < 2 or sebaran.min() < 10:
                raise ValueError("Setiap kelas sentimen harus punya minimal 10 ulasan. Sebaran label kamus: "
                                 + ", ".join(f"{k} {v}" for k, v in sebaran.items()) + ".")

            st.write("4/5 Pembagian data latih dan data uji (80:20)")
            data = P.bagi_data(data)
            if P.batas_gold(data)[1] < P.batas_gold(data)[0]:
                raise ValueError("Data uji terlalu sedikit untuk membentuk gold set (minimal 100 ulasan).")
            data["gold_dipilih"] = False

            st.write("5/5 Menyimpan sebagai data sementara")
            info = dict(nama_berkas=berkas.name, diunggah_oleh=auth.pengguna()["username"],
                        n_awal=ringkas["awal"], n_tidak_valid=ringkas["tidak_valid"],
                        n_duplikat=ringkas["duplikat"], n_akhir=len(data), n_gold=0,
                        n_kata_baru=len(kata_baru))
            dw.simpan_staging(engine, data, info)
            status.update(label="Pemrosesan selesai", state="complete", expanded=False)
        except ValueError as e:
            status.update(label="Pemrosesan dihentikan", state="error")
            st.error(str(e))
            return False
        except Exception as e:
            status.update(label="Pemrosesan gagal", state="error")
            st.error(f"Terjadi kesalahan tak terduga: {e}")
            return False
    st.session_state.pop("stg_kunci", None)
    st.session_state["pesan"] = ("Berkas berhasil diproses. Data di dashboard belum berubah; lanjutkan dengan "
                                 "menentukan gold set di bawah.")
    return True


st.title("Halaman Admin")
tab_proses, tab_gold, tab_akun = st.tabs(["1. Unggah dan proses data", "2. Gold set dan model", "3. Akun"])

# ============================== TAB 1 ==============================
with tab_proses:
    if "pesan" in st.session_state:
        st.success(st.session_state.pop("pesan"))

    # ---- Data yang sedang tampil di dashboard
    st.subheader("Data di dashboard saat ini")
    aktif = dw.ringkasan_aktif(engine)
    if aktif:
        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah ulasan", angka(aktif["n"]), border=True)
        c2.metric("Ulasan gold set", angka(aktif["n_gold"]), border=True)
        c3.metric("Rentang tanggal", f"{pd.to_datetime(aktif['dari']):%d/%m/%Y} - {pd.to_datetime(aktif['sampai']):%d/%m/%Y}", border=True)
    else:
        st.info("Belum ada data di dashboard.")

    # ---- Langkah 1: unggah
    st.subheader("Langkah 1. Unggah berkas hasil scraping")
    st.caption("Format .xlsx atau .csv dengan kolom: content, score, at, reviewCreatedVersion. "
               "Unggahan hanya membuat data sementara. Dashboard baru berubah setelah proses selesai dan "
               "Anda mengonfirmasinya di tab 2.")
    berkas = st.file_uploader("Berkas scraping", type=["xlsx", "csv"], label_visibility="collapsed")
    ada_sementara = dw.info_staging(engine) is not None
    if ada_sementara:
        st.caption("Sudah ada data sementara dari unggahan sebelumnya. Memproses berkas baru akan menggantinya.")
    if st.button("Proses berkas", type="primary", disabled=berkas is None):
        if proses_berkas(berkas):
            st.rerun()

    # ---- Langkah 2: hasil pemrosesan
    info = dw.info_staging(engine)
    if info:
        data = muat_data_sementara(info)
        st.subheader("Langkah 2. Hasil pemrosesan")
        st.caption(f"Berkas: {info['nama_berkas']} · diunggah oleh {info['diunggah_oleh']} · "
                   f"{wib(info['waktu'])}")
        dibuang_kosong = info["n_awal"] - info["n_tidak_valid"] - info["n_duplikat"] - info["n_akhir"]
        m = st.columns(5)
        m[0].metric("Ulasan awal", angka(info["n_awal"]), border=True)
        m[1].metric("Tidak valid", angka(info["n_tidak_valid"]), border=True)
        m[2].metric("Duplikat", angka(info["n_duplikat"]), border=True)
        m[3].metric("Teks kosong", angka(dibuang_kosong), border=True)
        m[4].metric("Ulasan dianalisis", angka(info["n_akhir"]), border=True)
        sebaran = data["label_kamus"].value_counts()
        m = st.columns(5)
        m[0].metric("Positif (kamus)", angka(sebaran.get("positif", 0)), border=True)
        m[1].metric("Netral (kamus)", angka(sebaran.get("netral", 0)), border=True)
        m[2].metric("Negatif (kamus)", angka(sebaran.get("negatif", 0)), border=True)
        m[3].metric("Data latih", angka((data["partisi"] == "latih").sum()), border=True)
        m[4].metric("Data uji", angka((data["partisi"] == "uji").sum()), border=True)
        st.caption(f"Kata baru yang di-stem pada proses ini: {angka(info['n_kata_baru'])}.")

        # ---- Langkah 3: gold set
        st.subheader("Langkah 3. Tentukan gold set")
        rendah, tinggi = P.batas_gold(data)
        n_uji = int((data["partisi"] == "uji").sum())
        awal = info["n_gold"] if info["n_gold"] else min(600, tinggi)
        n = st.number_input("Jumlah ulasan gold set", min_value=rendah, max_value=tinggi,
                            value=int(min(max(awal, rendah), tinggi)), step=50)
        moe = f"{margin_galat(n, n_uji):.1f}".replace(".", ",")
        st.caption(f"Diambil dari data uji ({angka(n_uji)} ulasan), berimbang menurut sebaran label kamus. "
                   f"Rentang yang diizinkan {angka(rendah)} - {angka(tinggi)}. "
                   f"Perkiraan margin galat akurasi dengan {angka(n)} ulasan: ±{moe} poin "
                   "persentase (kepercayaan 95%). Semakin banyak ulasan, semakin teliti hasil evaluasi, "
                   "tetapi semakin lama pelabelan manual.")
        ids = P.ambil_gold(data, int(n))
        komposisi = data[data["id"].isin(ids)]["label_kamus"].value_counts()
        st.caption("Perkiraan komposisi (menurut label kamus): "
                   + ", ".join(f"{k} {angka(komposisi.get(k, 0))}" for k in ("positif", "netral", "negatif"))
                   + ". Label sebenarnya Anda isi secara manual.")
        if st.button("Tetapkan gold set"):
            data["gold_dipilih"] = data["id"].isin(ids)
            info2 = {k: info[k] for k in ("nama_berkas", "diunggah_oleh", "n_awal", "n_tidak_valid",
                                          "n_duplikat", "n_akhir", "n_kata_baru")}
            info2["n_gold"] = len(ids)
            with st.spinner("Menyimpan gold set..."):
                dw.simpan_staging(engine, data, info2)
            st.session_state.pop("stg_kunci", None)
            st.session_state["pesan"] = (f"Gold set {angka(len(ids))} ulasan ditetapkan. Unduh lembarnya, isi "
                                         "kolom label_manual, lalu unggah di tab 2.")
            st.rerun()

        if info["n_gold"]:
            gold_ids = data.loc[data["gold_dipilih"], "id"].tolist()
            st.success(f"Gold set tersimpan: {angka(len(gold_ids))} ulasan.")
            st.download_button("Unduh lembar gold set (.xlsx)", data=P.lembar_gold(data, gold_ids),
                               file_name="gold_set_untuk_dilabel.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.caption("Isi kolom label_manual dengan positif, netral, atau negatif (tersedia pilihan dropdown). "
                       "Jangan menambah, menghapus, atau mengubah kolom id.")

# ============================== TAB 2 & 3 ==============================
with tab_gold:
    st.info("Unggah gold set berlabel, pelatihan model, evaluasi, dan penerapan ke dashboard akan "
            "ditambahkan di langkah berikutnya.")
with tab_akun:
    st.info("Pengelolaan akun akan ditambahkan di langkah berikutnya.")
