"""Akses data warehouse (PostgreSQL di Neon): tabel DW, kamus stem, data sementara (staging), ETL, cadangan."""
import os
import re
from datetime import datetime

import pandas as pd
from sqlalchemy import (Boolean, Column, Date, DateTime, ForeignKey, Integer, MetaData, SmallInteger,
                        String, Table, Text, delete, func, insert, select, text)

BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus",
         "September", "Oktober", "November", "Desember"]

meta = MetaData()
dim_sentimen = Table(
    "dim_sentimen", meta,
    Column("sentimen_key", Integer, primary_key=True, autoincrement=True),
    Column("label", String(10), nullable=False, unique=True))
dim_tanggal = Table(
    "dim_tanggal", meta,
    Column("tanggal_key", Integer, primary_key=True, autoincrement=False),   # YYYYMMDD
    Column("tanggal", Date, nullable=False),
    Column("tahun", Integer, nullable=False), Column("bulan", Integer, nullable=False),
    Column("nama_bulan", String(12), nullable=False), Column("tahun_bulan", String(7), nullable=False),
    Column("kuartal", Integer, nullable=False))
dim_versi = Table(
    "dim_versi", meta,
    Column("versi_key", Integer, primary_key=True, autoincrement=True),
    Column("versi_asli", String(30), nullable=False, unique=True),
    Column("versi_grup", String(30), nullable=False))
fact_ulasan = Table(
    "fact_ulasan", meta,
    Column("ulasan_id", Integer, primary_key=True, autoincrement=False),
    Column("tanggal_key", Integer, ForeignKey("dim_tanggal.tanggal_key"), nullable=False),
    Column("versi_key", Integer, ForeignKey("dim_versi.versi_key"), nullable=False),
    Column("rating", SmallInteger, nullable=False),
    Column("isi_ulasan", Text, nullable=False),
    Column("sentimen_nb_key", Integer, ForeignKey("dim_sentimen.sentimen_key"), nullable=False),
    Column("sentimen_rf_key", Integer, ForeignKey("dim_sentimen.sentimen_key"), nullable=False),
    Column("sentimen_manual_key", Integer, ForeignKey("dim_sentimen.sentimen_key")),   # hanya gold set
    Column("partisi", String(5), nullable=False),
    Column("is_gold", Boolean, nullable=False))
kamus_stem = Table(
    "kamus_stem", meta,
    Column("kata", Text, primary_key=True),
    Column("stem", Text, nullable=False))
staging_info = Table(
    "staging_info", meta,
    Column("id", Integer, primary_key=True, autoincrement=False),
    Column("nama_berkas", Text), Column("diunggah_oleh", String(30)), Column("waktu", DateTime),
    Column("n_awal", Integer), Column("n_tidak_valid", Integer), Column("n_duplikat", Integer),
    Column("n_akhir", Integer), Column("n_gold", Integer), Column("n_kata_baru", Integer))

KOLOM_STAGING = ["id", "content", "score", "at", "tanggal", "reviewCreatedVersion", "teks_prep",
                 "skor", "label_kamus", "partisi", "gold_dipilih"]


def _tulis(df, nama, con, **kw):
    df.to_sql(nama, con, index=False, method="multi", chunksize=2000, **kw)


# ============================ Persiapan tabel ============================
def siapkan(engine, csv_stem=None):
    """Buat tabel yang belum ada (yang sudah ada dibiarkan) dan isi kamus stem awal bila masih kosong."""
    meta.create_all(engine)
    if csv_stem and os.path.exists(csv_stem):
        with engine.begin() as c:
            if c.execute(select(func.count()).select_from(kamus_stem)).scalar() == 0:
                _tulis(pd.read_csv(csv_stem, keep_default_na=False), "kamus_stem", c, if_exists="append")


# ============================ Kamus stem ============================
def muat_peta_stem(engine):
    d = pd.read_sql(select(kamus_stem.c.kata, kamus_stem.c.stem), engine)
    return dict(zip(d["kata"], d["stem"]))


def simpan_kata_baru(engine, baru):
    if not baru:
        return
    kata = list(baru)
    with engine.begin() as c:
        ada = set()
        for i in range(0, len(kata), 5000):
            ada |= {r[0] for r in c.execute(select(kamus_stem.c.kata).where(kamus_stem.c.kata.in_(kata[i:i + 5000])))}
        tambah = [k for k in kata if k not in ada]
        if tambah:
            _tulis(pd.DataFrame({"kata": tambah, "stem": [baru[k] for k in tambah]}), "kamus_stem", c,
                   if_exists="append")


# ============================ Data sementara (staging) ============================
def simpan_staging(engine, df, info):
    """Simpan hasil pra-proses + label kamus + pembagian data + gold terpilih. Tidak menyentuh data dashboard."""
    with engine.begin() as c:
        _tulis(df[KOLOM_STAGING], "staging_ulasan", c, if_exists="replace")
        c.execute(delete(staging_info))
        c.execute(insert(staging_info).values(id=1, waktu=datetime.utcnow(), **info))


def info_staging(engine):
    with engine.connect() as c:
        r = c.execute(select(staging_info)).mappings().first()
    return dict(r) if r else None


def muat_staging(engine):
    d = pd.read_sql(text("select * from staging_ulasan order by id"), engine)
    d["at"] = pd.to_datetime(d["at"])
    d["tanggal"] = pd.to_datetime(d["tanggal"]).dt.date
    d["gold_dipilih"] = d["gold_dipilih"].astype(bool)
    return d


def hapus_staging(engine):
    with engine.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS staging_ulasan"))
        c.execute(delete(staging_info))


# ============================ Data aktif di dashboard ============================
def ringkasan_aktif(engine):
    """Ringkasan data yang sedang tampil di dashboard; None bila belum ada data."""
    q = text("""select count(*) as n, coalesce(sum(case when f.is_gold then 1 else 0 end), 0) as n_gold,
                       min(t.tanggal) as dari, max(t.tanggal) as sampai
                from fact_ulasan f join dim_tanggal t on f.tanggal_key = t.tanggal_key""")
    with engine.connect() as c:
        r = c.execute(q).mappings().first()
    return dict(r) if r and r["n"] else None


def ekspor_cadangan(engine):
    """Seluruh data aktif dalam satu tabel (untuk diunduh sebelum ditimpa)."""
    q = text("""select f.ulasan_id, t.tanggal, v.versi_asli as versi, f.rating, f.isi_ulasan,
                       nb.label as sentimen_nb, rf.label as sentimen_rf, m.label as sentimen_manual,
                       f.partisi, f.is_gold
                from fact_ulasan f
                join dim_tanggal t   on f.tanggal_key = t.tanggal_key
                join dim_versi v     on f.versi_key = v.versi_key
                join dim_sentimen nb on f.sentimen_nb_key = nb.sentimen_key
                join dim_sentimen rf on f.sentimen_rf_key = rf.sentimen_key
                left join dim_sentimen m on f.sentimen_manual_key = m.sentimen_key
                order by f.ulasan_id""")
    return pd.read_sql(q, engine)


# ============================ ETL: timpa data dashboard ============================
def _grup(v):
    m = re.match(r"^(\d+)\.(\d+)", v)
    return f"{m.group(1)}.{m.group(2)}" if m else "Tidak Diketahui"


def terapkan(engine, df, gold):
    """Ganti seluruh isi data warehouse dengan df (sudah berisi pred_nb, pred_rf) dan label manual gold.
    Berjalan dalam SATU transaksi: bila gagal di tengah, data lama di dashboard tetap utuh."""
    d = df.merge(gold, on="id", how="left")
    d["tanggal"] = pd.to_datetime(d["tanggal"])
    d["tanggal_key"] = d["tanggal"].dt.strftime("%Y%m%d").astype(int)
    d["reviewCreatedVersion"] = d["reviewCreatedVersion"].fillna("Tidak Diketahui").astype(str)
    d["content"] = d["content"].astype(str).str.replace("\x00", "", regex=False)

    sent = pd.DataFrame({"sentimen_key": [1, 2, 3], "label": ["positif", "netral", "negatif"]})
    tgl = pd.DataFrame({"tanggal": sorted(d["tanggal"].unique())})
    tgl["tanggal_key"] = tgl["tanggal"].dt.strftime("%Y%m%d").astype(int)
    tgl["tahun"] = tgl["tanggal"].dt.year
    tgl["bulan"] = tgl["tanggal"].dt.month
    tgl["nama_bulan"] = tgl["bulan"].apply(lambda b: BULAN[b - 1])
    tgl["tahun_bulan"] = tgl["tanggal"].dt.strftime("%Y-%m")
    tgl["kuartal"] = tgl["tanggal"].dt.quarter
    tgl["tanggal"] = tgl["tanggal"].dt.date
    ver = pd.DataFrame({"versi_asli": sorted(d["reviewCreatedVersion"].unique())})
    ver["versi_key"] = range(1, len(ver) + 1)
    ver["versi_grup"] = ver["versi_asli"].apply(_grup)

    peta_s = dict(zip(sent["label"], sent["sentimen_key"]))
    peta_v = dict(zip(ver["versi_asli"], ver["versi_key"]))
    fact = pd.DataFrame({
        "ulasan_id": d["id"], "tanggal_key": d["tanggal_key"],
        "versi_key": d["reviewCreatedVersion"].map(peta_v), "rating": d["score"], "isi_ulasan": d["content"],
        "sentimen_nb_key": d["pred_nb"].map(peta_s), "sentimen_rf_key": d["pred_rf"].map(peta_s),
        "sentimen_manual_key": d["label_manual"].map(peta_s).astype("Int64"),
        "partisi": d["partisi"], "is_gold": d["label_manual"].notna()})
    assert fact[["versi_key", "sentimen_nb_key", "sentimen_rf_key"]].notna().all().all(), "ada kunci yang tidak cocok"

    with engine.begin() as c:
        for t in (fact_ulasan, dim_tanggal, dim_versi, dim_sentimen):   # fakta dulu, lalu dimensi
            c.execute(delete(t))
        _tulis(sent, "dim_sentimen", c, if_exists="append")
        _tulis(tgl, "dim_tanggal", c, if_exists="append")
        _tulis(ver[["versi_key", "versi_asli", "versi_grup"]], "dim_versi", c, if_exists="append")
        _tulis(fact, "fact_ulasan", c, if_exists="append")
        # pemeriksaan di dalam transaksi; bila gagal, semuanya dibatalkan
        n = c.execute(select(func.count()).select_from(fact_ulasan)).scalar()
        n_gold = c.execute(select(func.count()).select_from(fact_ulasan).where(fact_ulasan.c.is_gold)).scalar()
        sql_rf = dict(c.execute(text("select s.label, count(*) from fact_ulasan f join dim_sentimen s "
                                     "on f.sentimen_rf_key = s.sentimen_key group by s.label")).all())
        if n != len(d) or n_gold != int(d["label_manual"].notna().sum()) or sql_rf != d["pred_rf"].value_counts().to_dict():
            raise RuntimeError("Pemeriksaan setelah pemuatan gagal; perubahan dibatalkan.")
    return {"ulasan": int(n), "gold": int(n_gold), "tanggal": len(tgl), "versi": len(ver)}
