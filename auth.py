"""Autentikasi: tabel users, hash kata sandi (bcrypt), daftar akun, masuk, dan penjaga peran."""
import os
import re
import time
from datetime import datetime

import bcrypt
import streamlit as st
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, func, insert, select
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
    """Dipanggil di awal setiap halaman admin; menghentikan halaman bila bukan admin."""
    u = pengguna()
    if not u or u["role"] != "admin":
        st.error("Halaman ini hanya untuk admin.")
        st.stop()


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
