import io
from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from scipy.stats import binomtest, chi2
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import auth
import dw
import pipeline as P
from db import get_engine

auth.wajib_admin()

BASE = Path(__file__).parent
engine = get_engine()
URUT = ["positif", "netral", "negatif"]
ALPHA = 0.05


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


def persen(x):
    return f"{x * 100:.1f}%".replace(".", ",")


def evaluasi(hasil, gold):
    """Metrik NB dan RF pada gold set (rata-rata makro atas kelas yang ada) dan uji McNemar."""
    g = hasil.merge(gold, on="id")
    kelas = [k for k in URUT if k in set(g["label_manual"])]
    metrik = {}
    for nama, kol in (("Naive Bayes", "pred_nb"), ("Random Forest", "pred_rf")):
        p, r, f, _ = precision_recall_fscore_support(g["label_manual"], g[kol], labels=kelas,
                                                     average="macro", zero_division=0)
        metrik[nama] = {"Accuracy": accuracy_score(g["label_manual"], g[kol]),
                        "Precision": p, "Recall": r, "F1-Score": f}
    nb_ok = g["label_manual"] == g["pred_nb"]
    rf_ok = g["label_manual"] == g["pred_rf"]
    b, c = int((nb_ok & ~rf_ok).sum()), int((~nb_ok & rf_ok).sum())
    n = b + c
    stat = None
    if n == 0:
        pval = 1.0
    elif n < 25:
        pval = binomtest(min(b, c), n, 0.5).pvalue
    else:
        stat = (abs(b - c) - 1) ** 2 / n
        pval = chi2.sf(stat, 1)
    if pval >= ALPHA:
        kesimpulan = "Tidak berbeda signifikan"
    else:
        kesimpulan = "Naive Bayes lebih baik" if b > c else "Random Forest lebih baik"
    return {"metrik": metrik, "b": b, "c": c, "stat": stat, "p": pval, "kesimpulan": kesimpulan, "n": len(g)}


def latih_dan_terapkan(data, gold):
    """Latih NB + RF, prediksi semua ulasan, evaluasi, lalu timpa data dashboard (satu transaksi)."""
    with st.status("Melatih model dan menerapkan ke dashboard...", expanded=True) as status:
        try:
            st.write("1/3 Pelatihan model (TF-IDF, Naive Bayes, Random Forest) dan prediksi semua ulasan")
            hasil, meta = P.latih_prediksi(data, lambda nama: st.write(f"· {nama}"))
            st.write("2/3 Evaluasi pada gold set dan uji McNemar")
            ev = evaluasi(hasil, gold)
            st.write("3/3 Memuat ke data warehouse (data lama ditimpa dalam satu transaksi)")
            ringkas = dw.terapkan(engine, hasil, gold)
            dw.hapus_staging(engine)
            st.cache_data.clear()   # dashboard memuat ulang data baru
            status.update(label="Selesai: dashboard sudah diperbarui", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Gagal; data di dashboard tidak berubah", state="error")
            st.error(f"Proses dibatalkan: {e}")
            return False
    for k in ("stg_kunci", "stg_df", "cadangan", "cadangan_diunduh"):
        st.session_state.pop(k, None)
    st.session_state["hasil_terapan"] = {"ev": ev, "ringkas": ringkas, "fitur": meta["fitur"],
                                         "nb": hasil["pred_nb"].value_counts().to_dict(),
                                         "rf": hasil["pred_rf"].value_counts().to_dict()}
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
    hasil = st.session_state.get("hasil_terapan")
    if hasil:
        ev, rg = hasil["ev"], hasil["ringkas"]
        st.success(f"Data di dashboard sudah diperbarui: {angka(rg['ulasan'])} ulasan, "
                   f"{angka(rg['gold'])} di antaranya gold set. Buka menu Dashboard untuk melihat hasilnya.")
        st.subheader("Ringkasan evaluasi pada gold set")
        tabel = pd.DataFrame({n: {k: persen(v) for k, v in m.items()} for n, m in ev["metrik"].items()}).T
        st.dataframe(tabel, use_container_width=True)
        stat = "tidak dihitung (uji eksak)" if ev["stat"] is None else f"{ev['stat']:.3f}".replace(".", ",")
        st.caption(f"Uji McNemar: b = {ev['b']} (NB benar, RF salah), c = {ev['c']} (NB salah, RF benar), "
                   f"chi-square {stat}, p = {ev['p']:.4f}".replace(".", ",") + f". Kesimpulan: {ev['kesimpulan']} "
                   f"(α = 0,05). Jumlah fitur TF-IDF: {angka(hasil['fitur'])}.")
        if st.button("Tutup ringkasan"):
            st.session_state.pop("hasil_terapan")
            st.rerun()
        st.divider()

    info_s = dw.info_staging(engine)
    if not info_s or not info_s["n_gold"]:
        st.info("Belum ada gold set yang menunggu dilabel. Selesaikan tab 1: proses berkas lalu tetapkan gold set.")
    else:
        stg = muat_data_sementara(info_s)
        gold_ids = stg.loc[stg["gold_dipilih"], "id"].tolist()
        st.subheader("Langkah 4. Unggah gold set yang sudah dilabel")
        st.caption(f"Gunakan lembar yang diunduh dari tab 1 ({angka(len(gold_ids))} ulasan) dengan kolom "
                   "label_manual terisi penuh.")
        lembar = st.file_uploader("Gold set berlabel", type=["xlsx"], key="up_gold", label_visibility="collapsed")
        gold = None
        if lembar is not None:
            try:
                gold = P.baca_gold(io.BytesIO(lembar.getvalue()), gold_ids)
            except ValueError as e:
                st.error(str(e))
        if gold is not None:
            s = gold["label_manual"].value_counts()
            st.success(f"Gold set valid: {angka(len(gold))} ulasan. Positif {angka(s.get('positif', 0))}, "
                       f"netral {angka(s.get('netral', 0))}, negatif {angka(s.get('negatif', 0))}.")

            st.subheader("Langkah 5. Latih model dan terapkan ke dashboard")
            aktif2 = dw.ringkasan_aktif(engine)
            siap = True
            if aktif2:
                st.warning(f"Data di dashboard saat ini ({angka(aktif2['n'])} ulasan) akan DITIMPA oleh data baru "
                           f"({angka(info_s['n_akhir'])} ulasan). Unduh cadangannya terlebih dahulu.")
                if st.button("Siapkan cadangan data saat ini"):
                    with st.spinner("Menyiapkan cadangan..."):
                        st.session_state["cadangan"] = dw.ekspor_cadangan(engine).to_csv(
                            index=False).encode("utf-8-sig")
                    st.session_state.pop("cadangan_diunduh", None)
                if "cadangan" in st.session_state:
                    st.download_button(
                        "Unduh cadangan (.csv)", data=st.session_state["cadangan"],
                        file_name="cadangan_data_dashboard.csv", mime="text/csv",
                        on_click=lambda: st.session_state.update(cadangan_diunduh=True))
                yakin = st.checkbox("Saya mengerti data di dashboard akan ditimpa")
                siap = yakin and st.session_state.get("cadangan_diunduh", False)
                if not siap:
                    st.caption("Tombol aktif setelah cadangan diunduh dan kotak persetujuan dicentang.")
            if st.button("Latih model dan terapkan", type="primary", disabled=not siap):
                if latih_dan_terapkan(stg, gold):
                    st.rerun()

        with st.expander("Buang data sementara"):
            st.caption("Menghapus hasil pemrosesan dan gold set yang belum diterapkan. Data di dashboard "
                       "tidak terpengaruh.")
            buang = st.checkbox("Ya, buang data sementara")
            if st.button("Buang", disabled=not buang):
                dw.hapus_staging(engine)
                for k in ("stg_kunci", "stg_df", "cadangan", "cadangan_diunduh"):
                    st.session_state.pop(k, None)
                st.rerun()
with tab_akun:
    st.info("Pengelolaan akun akan ditambahkan di langkah berikutnya.")
