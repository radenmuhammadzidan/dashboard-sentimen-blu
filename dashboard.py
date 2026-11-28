from pathlib import Path

import streamlit as st

import auth
import dw
from db import get_engine

st.set_page_config(page_title="Dashboard NB vs RF", layout="wide")


@st.cache_resource
def siapkan_sekali():
    mesin = get_engine()
    auth.siapkan(mesin)   # buat tabel users + admin pertama (dari Secrets) bila belum ada
    dw.siapkan(mesin, Path(__file__).parent / "kamus" / "stem_cache.csv")   # tabel DW, kamus stem, data sementara
    return True


siapkan_sekali()

akun = auth.pengguna()
if akun is None:
    auth.tampil_login(get_engine())
    st.stop()

# Halaman admin hanya didaftarkan untuk admin (dan dijaga lagi di dalam halamannya)
halaman = [st.Page("halaman_dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True)]
if akun["role"] == "admin":
    halaman.append(st.Page("halaman_admin.py", title="Admin", icon=":material/admin_panel_settings:"))
nav = st.navigation(halaman)

with st.sidebar:
    st.markdown(f"Masuk sebagai **{akun['username']}**  \n"
                f"Peran: {'Admin' if akun['role'] == 'admin' else 'Pengguna'}")
    if st.button("Keluar", use_container_width=True):
        auth.keluar()
        st.rerun()

nav.run()
