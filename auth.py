"""Autentikasi: tabel users, hash kata sandi (bcrypt), daftar akun, masuk, dan penjaga peran."""
import os
import re
import time
from datetime import datetime

import bcrypt
import streamlit as st
import pandas as pd
from sqlalchemy import (Column, DateTime, Integer, MetaData, String, Table, delete, func, insert, select,
                        update)
from sqlalchemy.exc import IntegrityError

meta = MetaData()
users = Table(
    "users", meta,
    Column("user_id", Integer, primary_key=True, autoincrement=True),
    Column("username", String(30), unique=True, nullable=False),
    Column("password_hash", String(100), nullable=False),
    Column("role", String(10), nullable=False),          # "admin" atau "user"
    Column("dibuat", DateTime, nullable=False),
)

PASS_MIN, PASS_MAKS = 8, 64          # bcrypt hanya membaca 72 byte pertama
MAKS_GAGAL, JEDA_DETIK = 5, 60       # batas percobaan masuk per sesi
_HASH_PALSU = bcrypt.hashpw(b"palsu", bcrypt.gensalt()).decode()   # penyamar waktu cek saat username tidak ada


def hash_pw(pw):
    return bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt()).decode()


def cek_pw(pw, hashed):
    try:
        return bcrypt.checkpw(pw.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


def siapkan(engine):
    """Buat tabel users bila belum ada, lalu buat admin pertama dari Secrets bila belum ada admin."""
    meta.create_all(engine, tables=[users])
    nama = (os.getenv("ADMIN_USERNAME") or "").strip().lower()
    pw = os.getenv("ADMIN_PASSWORD") or ""
    if not nama or not pw:
        return
    with engine.begin() as c:
        ada = c.execute(select(func.count()).select_from(users).where(users.c.role == "admin")).scalar()
        if ada == 0:
            c.execute(insert(users).values(username=nama, password_hash=hash_pw(pw),
                                           role="admin", dibuat=datetime.utcnow()))


def daftar(engine, username, password, ulang):
    """Pendaftaran publik: peran SELALU 'user'. Mengembalikan (berhasil, pesan)."""
    username = (username or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9_]{3,30}", username):
        return False, "Username 3-30 karakter: huruf, angka, atau garis bawah."
    if not (PASS_MIN <= len(password) <= PASS_MAKS) or len(password.encode("utf-8")) > 72:
        return False, f"Kata sandi {PASS_MIN}-{PASS_MAKS} karakter."
    if password != ulang:
        return False, "Kata sandi dan pengulangannya tidak sama."
    try:
        with engine.begin() as c:
            c.execute(insert(users).values(username=username, password_hash=hash_pw(password),
                                           role="user", dibuat=datetime.utcnow()))
    except IntegrityError:
        return False, "Username sudah dipakai."
    return True, "Akun berhasil dibuat. Silakan masuk."


def masuk(engine, username, password):
    """Mengembalikan {'username', 'role'} bila cocok, selain itu None."""
    username = (username or "").strip().lower()
    with engine.connect() as c:
        r = c.execute(select(users.c.username, users.c.password_hash, users.c.role)
                      .where(users.c.username == username)).first()
    if r is None:
        cek_pw(password or "", _HASH_PALSU)   # waktu proses tetap sama
        return None
    if cek_pw(password or "", r.password_hash):
        return {"username": r.username, "role": r.role}
    return None


def pengguna():
    return st.session_state.get("auth_user")


def keluar():
    st.session_state.pop("auth_user", None)


def wajib_admin():
    """Dipanggil di awal setiap halaman admin; menghentikan halaman bila bukan admin.
    Peran diperiksa ulang ke database, sehingga akun yang diturunkan atau dihapus kehilangan akses
    tanpa menunggu keluar."""
    u = pengguna()
    if not u or u["role"] != "admin":
        st.error("Halaman ini hanya untuk admin.")
        st.stop()
    from db import get_engine
    with get_engine().connect() as c:
        peran = c.execute(select(users.c.role).where(users.c.username == u["username"])).scalar()
    if peran is None:
        keluar()
        st.rerun()
    if peran != "admin":
        st.session_state["auth_user"] = {"username": u["username"], "role": peran}
        st.rerun()


# ============================== Pengelolaan akun (khusus admin) ==============================
def daftar_akun(engine):
    d = pd.read_sql(select(users.c.username, users.c.role, users.c.dibuat).order_by(users.c.dibuat), engine)
    return d


def _jumlah_admin(c):
    return c.execute(select(func.count()).select_from(users).where(users.c.role == "admin")).scalar()


def ubah_peran(engine, username, peran_baru, oleh):
    if username == oleh:
        return False, "Peran akun sendiri tidak dapat diubah."
    if peran_baru not in ("admin", "user"):
        return False, "Peran tidak dikenal."
    with engine.begin() as c:
        peran = c.execute(select(users.c.role).where(users.c.username == username)).scalar()
        if peran is None:
            return False, "Akun tidak ditemukan."
        if peran == "admin" and peran_baru != "admin" and _jumlah_admin(c) <= 1:
            return False, "Harus tersisa minimal satu admin."
        c.execute(update(users).where(users.c.username == username).values(role=peran_baru))
    return True, f"Peran {username} diubah menjadi {'Admin' if peran_baru == 'admin' else 'Pengguna'}."


def hapus_akun(engine, username, oleh):
    if username == oleh:
        return False, "Akun sendiri tidak dapat dihapus."
    with engine.begin() as c:
        peran = c.execute(select(users.c.role).where(users.c.username == username)).scalar()
        if peran is None:
            return False, "Akun tidak ditemukan."
        if peran == "admin" and _jumlah_admin(c) <= 1:
            return False, "Harus tersisa minimal satu admin."
        c.execute(delete(users).where(users.c.username == username))
    return True, f"Akun {username} dihapus."


def atur_ulang_sandi(engine, username, sandi, ulang):
    if not (PASS_MIN <= len(sandi) <= PASS_MAKS) or len(sandi.encode("utf-8")) > 72:
        return False, f"Kata sandi {PASS_MIN}-{PASS_MAKS} karakter."
    if sandi != ulang:
        return False, "Kata sandi dan pengulangannya tidak sama."
    with engine.begin() as c:
        hasil = c.execute(update(users).where(users.c.username == username).values(password_hash=hash_pw(sandi)))
    if hasil.rowcount == 0:
        return False, "Akun tidak ditemukan."
    return True, f"Kata sandi {username} diganti."


def tampil_kelola_akun(engine):
    """Isi tab Akun di halaman admin."""
    saya = pengguna()["username"]
    if "pesan_akun" in st.session_state:
        st.success(st.session_state.pop("pesan_akun"))
    d = daftar_akun(engine)
    st.subheader("Daftar akun")
    tampil = pd.DataFrame({
        "Username": d["username"],
        "Peran": d["role"].map({"admin": "Admin", "user": "Pengguna"}),
        "Dibuat": (pd.to_datetime(d["dibuat"]) + pd.Timedelta(hours=7)).dt.strftime("%d/%m/%Y %H:%M WIB")})
    st.dataframe(tampil, hide_index=True, use_container_width=True)
    st.caption(f"{len(d)} akun, {int((d['role'] == 'admin').sum())} di antaranya admin. "
               "Pendaftaran publik selalu berperan Pengguna; hanya admin yang dapat menaikkan peran.")

    lain = [u for u in d["username"] if u != saya]

    st.subheader("Ubah peran")
    if lain:
        with st.form("form_peran"):
            akun = st.selectbox("Akun", lain)
            peran = st.radio("Peran baru", ["Pengguna", "Admin"], horizontal=True)
            if st.form_submit_button("Simpan peran"):
                ok, pesan = ubah_peran(engine, akun, "admin" if peran == "Admin" else "user", saya)
                if ok:
                    st.session_state["pesan_akun"] = pesan
                    st.rerun()
                st.error(pesan)
    else:
        st.caption("Belum ada akun lain.")

    st.subheader("Atur ulang kata sandi")
    with st.form("form_sandi", clear_on_submit=True):
        akun2 = st.selectbox("Akun", d["username"].tolist(), key="sandi_akun")
        s1 = st.text_input("Kata sandi baru", type="password")
        s2 = st.text_input("Ulangi kata sandi baru", type="password")
        if st.form_submit_button("Ganti kata sandi"):
            ok, pesan = atur_ulang_sandi(engine, akun2, s1, s2)
            if ok:
                st.session_state["pesan_akun"] = pesan
                st.rerun()
            st.error(pesan)

    st.subheader("Hapus akun")
    if lain:
        with st.form("form_hapus"):
            akun3 = st.selectbox("Akun", lain, key="hapus_akun")
            yakin = st.checkbox("Ya, hapus akun ini secara permanen")
            if st.form_submit_button("Hapus akun"):
                if not yakin:
                    st.error("Centang kotak konfirmasi terlebih dahulu.")
                else:
                    ok, pesan = hapus_akun(engine, akun3, saya)
                    if ok:
                        st.session_state["pesan_akun"] = pesan
                        st.rerun()
                    st.error(pesan)
    else:
        st.caption("Belum ada akun lain.")


def tampil_login(engine):
    """Halaman Masuk / Daftar. Mengisi st.session_state['auth_user'] saat berhasil masuk."""
    st.markdown(
        "<style>.stApp{background:#eef2f9;} .st-key-login{background:#fff;border-radius:18px;"
        "padding:26px 28px;box-shadow:0 2px 10px rgba(20,40,90,.07);}"
        ".lg-h{background:linear-gradient(90deg,#1565d8,#2f86ec);border-radius:18px;padding:20px 26px;"
        "color:#fff;font-size:1.3rem;font-weight:700;margin-bottom:16px;}"
        ".lg-h small{display:block;font-size:.82rem;font-weight:400;color:#d6e6ff;}</style>",
        unsafe_allow_html=True)
    _, tengah, _ = st.columns([1, 2, 1])
    with tengah:
        st.markdown('<div class="lg-h">Dashboard Perbandingan Naive Bayes vs Random Forest'
                    '<small>blu by BCA Digital · Silakan masuk untuk melanjutkan</small></div>',
                    unsafe_allow_html=True)
        with st.container(key="login"):
            tab_masuk, tab_daftar = st.tabs(["Masuk", "Daftar akun"])
            with tab_masuk:
                with st.form("form_masuk"):
                    nama = st.text_input("Username")
                    sandi = st.text_input("Kata sandi", type="password")
                    kirim = st.form_submit_button("Masuk", type="primary", use_container_width=True)
                if kirim:
                    gagal = st.session_state.get("auth_gagal", 0)
                    tunggu = st.session_state.get("auth_tunggu", 0) - time.time()
                    if gagal >= MAKS_GAGAL and tunggu > 0:
                        st.error(f"Terlalu banyak percobaan. Coba lagi dalam {int(tunggu) + 1} detik.")
                    else:
                        if gagal >= MAKS_GAGAL:
                            gagal = 0
                        akun = masuk(engine, nama, sandi)
                        if akun:
                            st.session_state["auth_user"] = akun
                            st.session_state["auth_gagal"] = 0
                            st.rerun()
                        else:
                            gagal += 1
                            st.session_state["auth_gagal"] = gagal
                            if gagal >= MAKS_GAGAL:
                                st.session_state["auth_tunggu"] = time.time() + JEDA_DETIK
                            st.error("Username atau kata sandi salah.")
            with tab_daftar:
                st.caption("Akun yang dibuat di sini berperan sebagai pengguna: dapat melihat dashboard dan "
                           "mengubah filter.")
                with st.form("form_daftar"):
                    n2 = st.text_input("Username baru")
                    p2 = st.text_input("Kata sandi baru", type="password")
                    p3 = st.text_input("Ulangi kata sandi", type="password")
                    kirim2 = st.form_submit_button("Buat akun", use_container_width=True)
                if kirim2:
                    ok, pesan = daftar(engine, n2, p2, p3)
                    (st.success if ok else st.error)(pesan)
