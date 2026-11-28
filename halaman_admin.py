import streamlit as st

import auth

auth.wajib_admin()

st.title("Halaman Admin")
st.info("Unggah data, gold set, dan pengelolaan akun akan ditambahkan di langkah berikutnya.")
