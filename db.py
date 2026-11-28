import os
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

load_dotenv()


@st.cache_resource
def get_engine():
    return create_engine(
        URL.create(
            "postgresql+psycopg2",
            username=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT")),
            database=os.getenv("DB_NAME"),
        ),
        pool_pre_ping=True,    # cek koneksi sebelum dipakai; koneksi basi (Neon menutup koneksi yang menganggur) diganti otomatis
        pool_recycle=240,      # buang koneksi yang lebih tua dari 4 menit
        connect_args={"connect_timeout": 20},   # beri waktu bila database baru "bangun" dari tidur
    )
