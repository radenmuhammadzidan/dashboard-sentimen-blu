import altair as alt
import pandas as pd
import streamlit as st
from scipy.stats import binomtest, chi2
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from db import get_engine

URUT = ["positif", "negatif", "netral"]   # urutan tampil, mengikuti Gambar 2.4
MODEL = ["Naive Bayes", "Random Forest"]
RGB = {"Naive Bayes": (47, 111, 214), "Random Forest": (31, 157, 107)}
WARNA_DIST = {"Positif": "#1f9d6b", "Negatif": "#6fa8e8", "Netral": "#f0a020"}
METRIK = ["Accuracy", "Precision", "Recall", "F1-Score"]
SINGKAT = {"Naive Bayes": "NB", "Random Forest": "RF"}
ALPHA = 0.05
N_MIN = 30   # di bawah ini, hasil dianggap kurang stabil

CSS = """
<style>
.stApp, [data-testid="stAppViewContainer"] {background:#eef2f9 !important;}
.block-container {max-width: 100% !important; padding: 1.5rem 2.5rem 2rem 2.5rem;}
header[data-testid="stHeader"] {background: transparent;}

/* reset garis/kotak bawaan Streamlit pada tabel HTML */
[data-testid="stMarkdownContainer"] table {border:none !important; margin:0 !important;}
[data-testid="stMarkdownContainer"] tr {border:none !important; background:transparent !important;}
[data-testid="stMarkdownContainer"] th, [data-testid="stMarkdownContainer"] td {border:none !important; background:transparent;}

.st-key-hdr {background: linear-gradient(90deg,#1565d8,#2f86ec); border-radius: 18px; padding: 22px 28px;}
[class*="st-key-card_"] {background:#fff; border-radius:18px; padding:22px 24px; box-shadow:0 2px 10px rgba(20,40,90,.07);}
.st-key-card_metode {background:#0b3a7a !important; color:#fff;}
.st-key-card_cm_nb, .st-key-card_cm_rf, .st-key-card_dist {min-height: 600px;}
.st-key-flt {background: linear-gradient(90deg,#1565d8,#2f86ec); border-radius:14px; padding:14px 16px; margin-bottom:14px;}
.st-key-flt label p, .st-key-flt [data-testid="stWidgetLabel"] p {color:#fff !important; font-size:.78rem; font-weight:600;}
.st-key-flt [data-baseweb="tag"] {background:#dce8fb !important;}
.st-key-flt [data-baseweb="tag"] span {color:#1565d8 !important;}

.fi {background:#fff; border-radius:12px; padding:10px 18px; font-size:.82rem; color:#3b4a60; box-shadow:0 2px 10px rgba(20,40,90,.07); line-height:1.6;}
.fi b {color:#1565d8;}
.fw {background:#fff1d6; color:#9a6200; border-radius:12px; padding:9px 18px; font-size:.82rem; font-weight:600; margin-top:8px;}
.hd {display:flex; align-items:center; gap:14px;}
.logo {background:#fff; color:#1565d8; font-weight:800; border-radius:12px; padding:8px 16px; font-size:1.4rem; letter-spacing:-.02em;}
.hd-t {color:#fff; font-size:1.5rem; font-weight:700; line-height:1.2;}
.hd-s {color:#d6e6ff; font-size:.85rem;}

.ct {display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:14px;}
.ct-t {font-size:1.1rem; font-weight:700; color:#1a2333;}
.ct-s {font-size:.78rem; color:#6b7a90;}
.ct-r {display:flex; gap:8px; align-items:center;}
.bd {font-size:.75rem; font-weight:700; color:#1a2333;}
.pill {font-size:.75rem; font-weight:700; padding:3px 10px; border-radius:999px;}
.pill-nb {background:#dce8fb; color:#2f6fd6;}
.pill-rf {background:#d8f1e6; color:#1f9d6b;}

.mt-top {display:flex; justify-content:space-between; align-items:center;}
.mt-l {font-size:.75rem; letter-spacing:.08em; color:#b9d0f2;}
.mt-b {font-size:.72rem; font-weight:700; color:#6ee7b7; border:1px solid #34d399; border-radius:999px; padding:3px 12px;}
.mt-mid {display:flex; align-items:center; margin:2px 0 8px 0;}
.mt-sp {flex:1;}
.mt-name {flex:2; text-align:center; font-size:2rem; font-weight:700;}
.mt-val {flex:1; text-align:right; font-size:1.6rem; font-weight:700; color:#6ee7b7;}
.mt-val small {display:block; font-size:.7rem; font-weight:400; color:#b9d0f2;}
.mt-txt {font-size:.95rem; color:#fff;}
.mt-txt2 {font-size:.82rem; color:#b9d0f2; margin-top:2px;}
.mt-st {font-size:.78rem; font-weight:700; margin-top:10px; color:#b9d0f2;}

table.pf {width:100%; table-layout:fixed; border-collapse:collapse;}
table.pf th {font-size:.68rem; letter-spacing:.06em; color:#6b7a90; font-weight:600; text-align:center; padding:10px 4px;}
table.pf th.l, table.pf td.l {text-align:left; width:34%;}
table.pf td {text-align:center; padding:14px 4px; border-bottom:1px solid #eef1f6 !important;}
table.pf tr.best-row td {background:#eef8f2 !important;}
.tag {display:inline-block; width:30px; height:30px; line-height:30px; text-align:center; border-radius:8px; font-size:.7rem; font-weight:800; margin:0 10px 0 6px;}
.tag-nb {background:#dce8fb; color:#2f6fd6;}
.tag-rf {background:#d8f1e6; color:#1f9d6b;}
.mn {display:inline-block; vertical-align:middle; font-size:.9rem;}
.mb {font-size:.68rem; color:#1f9d6b; font-weight:600;}
.pl {display:inline-block; background:#f0f3f8; color:#1a2333; border-radius:8px; padding:5px 10px; font-size:.85rem; font-weight:600;}
.pl.top {background:#d8f1e6; color:#17794f;}

.sec {font-weight:700; font-size:.78rem; margin:20px 0 6px 0; color:#6b7a90; letter-spacing:.04em;}
table.pk {width:100%; table-layout:fixed; border-collapse:collapse; font-size:.85rem; color:#3b4a60;}
table.pk th {text-align:center; color:#6b7a90; font-weight:600; font-size:.72rem; padding:6px 4px;}
table.pk td {text-align:center; padding:9px 4px; border-bottom:1px solid #eef1f6 !important;}
table.pk th:first-child, table.pk td:first-child {text-align:left; width:26%;}

.mc {border:1px solid #e2e8f2; border-radius:12px; padding:16px 18px;}
.mc-t {font-weight:700; font-size:.95rem; margin-bottom:10px; color:#1a2333;}
.kv {display:flex; justify-content:space-between; font-size:.85rem; padding:5px 0; color:#3b4a60; border-bottom:1px solid #f1f4f9;}
.kv b {color:#1a2333;}
.dec {margin-top:12px; text-align:center; font-weight:700; font-size:.85rem; border-radius:8px; padding:10px;}
.dec.ok {background:#d8f1e6; color:#17794f;}
.dec.no {background:#fff1d6; color:#9a6200;}
.nt {font-size:.75rem; color:#6b7a90; margin-top:10px; line-height:1.4;}

.pm {text-align:center; font-size:.68rem; letter-spacing:.06em; color:#6b7a90; margin-bottom:4px;}
table.cmt {width:100%; table-layout:fixed; border-collapse:separate !important; border-spacing:5px;}
table.cmt th {font-size:.72rem; color:#6b7a90; font-weight:600; text-align:center; padding:2px 0;}
table.cmt th:nth-child(4), table.cmt td:nth-child(4) {width:58px;}
.cc {width:100%; height:46px; line-height:46px; text-align:center; border-radius:8px; font-weight:700; font-size:.95rem;}
td.rl, th.rl {font-size:.72rem; color:#6b7a90; text-align:left; padding-left:4px;}
.lg {display:flex; align-items:center; gap:8px; font-size:.68rem; color:#6b7a90; margin-top:10px;}
.lg span {display:inline-block; width:70px; height:6px; border-radius:3px;}
.dn {font-weight:700; font-size:.85rem; color:#1a2333; margin-bottom:2px;}
.dc {font-size:.78rem; color:#6b7a90; margin-top:6px;}
.mt-li {font-size:.9rem; color:#e6f0ff; margin-top:8px; line-height:1.5;}
.mt-ar {font-size:.78rem; color:#b9d0f2; margin-top:10px; line-height:1.45;}
</style>
"""


@st.cache_data
def muat_gold():
    q = """select f.ulasan_id, t.tanggal, v.versi_grup,
                  m.label as aktual, nb.label as pred_nb, rf.label as pred_rf
           from fact_ulasan f
           join dim_tanggal t   on f.tanggal_key = t.tanggal_key
           join dim_versi v     on f.versi_key = v.versi_key
           join dim_sentimen m  on f.sentimen_manual_key = m.sentimen_key
           join dim_sentimen nb on f.sentimen_nb_key = nb.sentimen_key
           join dim_sentimen rf on f.sentimen_rf_key = rf.sentimen_key
           where f.is_gold"""
    d = pd.read_sql(q, get_engine())
    d["tanggal"] = pd.to_datetime(d["tanggal"]).dt.date
    return d


@st.cache_data
def muat_semua():
    q = """select f.ulasan_id, t.tanggal, v.versi_grup,
                  nb.label as pred_nb, rf.label as pred_rf
           from fact_ulasan f
           join dim_tanggal t   on f.tanggal_key = t.tanggal_key
           join dim_versi v     on f.versi_key = v.versi_key
           join dim_sentimen nb on f.sentimen_nb_key = nb.sentimen_key
           join dim_sentimen rf on f.sentimen_rf_key = rf.sentimen_key"""
    d = pd.read_sql(q, get_engine())
    d["tanggal"] = pd.to_datetime(d["tanggal"]).dt.date
    return d


def hitung(gold, kol, labels=URUT):
    # rata-rata makro hanya atas kelas yang punya data pada gold set (labels)
    p, r, f, _ = precision_recall_fscore_support(
        gold["aktual"], gold[kol], labels=labels, average="macro", zero_division=0)
    return {"Accuracy": accuracy_score(gold["aktual"], gold[kol]),
            "Precision": p, "Recall": r, "F1-Score": f}


def mcnemar(sub):
    nb_ok = sub["aktual"] == sub["pred_nb"]
    rf_ok = sub["aktual"] == sub["pred_rf"]
    b = int((nb_ok & ~rf_ok).sum())   # NB benar, RF salah
    c = int((~nb_ok & rf_ok).sum())   # NB salah, RF benar
    n = b + c
    stat = None
    if n == 0:
        p = 1.0
    elif n < 25:
        p = binomtest(min(b, c), n, 0.5).pvalue          # uji eksak
    else:
        stat = (abs(b - c) - 1) ** 2 / n                 # chi-square + koreksi kontinuitas
        p = chi2.sf(stat, 1)
    if p >= ALPHA:
        unggul = "Tidak berbeda signifikan"
    else:
        unggul = "Naive Bayes" if b > c else "Random Forest"
    return b, c, p, unggul, stat


def urut_versi(v):
    try:
        return (0, [int(x) for x in v.split(".")])
    except ValueError:
        return (1, [0])   # "Tidak Diketahui" di paling akhir


def fmt(n):
    return f"{n:,}".replace(",", ".")


def pct(x):
    return f"{x * 100:.1f}".replace(".", ",") + "%"


def p_txt(p):
    return "&lt; 0,001" if p < 0.001 else f"{p:.4f}".replace(".", ",")


# ---------- potongan HTML (satu baris, tanpa baris kosong) ----------
def judul_kartu(judul, sub, badge, pill="", kls=""):
    pill_html = f'<span class="pill {kls}">{pill}</span>' if pill else ""
    return (f'<div class="ct"><div><div class="ct-t">{judul}</div><div class="ct-s">{sub}</div></div>'
            f'<div class="ct-r"><span class="bd">{badge}</span>{pill_html}</div></div>')


def html_perf(perf, terbaik):
    best = {m: perf[m].idxmax() for m in METRIK}
    head = "<tr><th class='l'>MODEL</th>" + "".join(f"<th>{m.upper()}</th>" for m in METRIK) + "</tr>"
    rows = ""
    for model in MODEL:
        baris_kls = "best-row" if model == terbaik else ""
        sub = '<div class="mb">Metode terbaik</div>' if model == terbaik else ""
        sel = ""
        for m in METRIK:
            top = "top" if best[m] == model else ""
            sel += f'<td><span class="pl {top}">{pct(perf.loc[model, m])}</span></td>'
        tag = SINGKAT[model].lower()
        rows += (f'<tr class="{baris_kls}"><td class="l"><span class="tag tag-{tag}">{SINGKAT[model]}</span>'
                 f'<span class="mn"><b>{model}</b>{sub}</span></td>{sel}</tr>')
    return f'<table class="pf">{head}{rows}</table>'


def html_mc_kelas(gold):
    tabel = "<tr><th>KELAS</th><th>b</th><th>c</th><th>P-VALUE</th><th>UNGGUL</th></tr>"
    for k in URUT:
        sub = gold[gold["aktual"] == k]
        if sub.empty:
            tabel += f"<tr><td>{k.capitalize()}</td><td>–</td><td>–</td><td>–</td><td><b>–</b></td></tr>"
            continue
        bk, ck, pk, uk, _ = mcnemar(sub)
        u = SINGKAT.get(uk, "–")
        tabel += (f"<tr><td>{k.capitalize()}</td><td>{bk}</td><td>{ck}</td>"
                  f"<td>{p_txt(pk)}</td><td><b>{u}</b></td></tr>")
    return (f'<div class="sec">MCNEMAR PER KELAS (berdasarkan label manual)</div>'
            f'<table class="pk">{tabel}</table>')


def html_mc_total(gold):
    b, c, p, unggul, stat = mcnemar(gold)
    ok_nb = int((gold["aktual"] == gold["pred_nb"]).sum())
    ok_rf = int((gold["aktual"] == gold["pred_rf"]).sum())
    if stat is None:
        baris_stat = '<div class="kv"><span>Uji</span><b>Binomial eksak</b></div>'
    else:
        baris_stat = f'<div class="kv"><span>χ² (koreksi kontinuitas)</span><b>{stat:.2f}</b></div>'
    if p < ALPHA:
        dec = f'<div class="dec ok">Keputusan: berbeda signifikan ({unggul} lebih baik)</div>'
        cat = "Selisih jumlah prediksi benar antara kedua model bukan kebetulan."
    else:
        dec = '<div class="dec no">Keputusan: tidak berbeda signifikan</div>'
        cat = "Selisih jumlah prediksi benar antara kedua model kecil dan dapat terjadi karena kebetulan."
    return (f'<div class="mc"><div class="mc-t">Uji signifikansi McNemar (NB vs RF)</div>'
            f'<div class="kv"><span>Prediksi benar · NB</span><b>{ok_nb} dari {len(gold)}</b></div>'
            f'<div class="kv"><span>Prediksi benar · RF</span><b>{ok_rf} dari {len(gold)}</b></div>'
            f'<div class="kv"><span>b · NB benar, RF salah</span><b>{b}</b></div>'
            f'<div class="kv"><span>c · RF benar, NB salah</span><b>{c}</b></div>'
            f'{baris_stat}'
            f'<div class="kv"><span>p-value (α = {str(ALPHA).replace(".", ",")})</span><b>{p_txt(p)}</b></div>'
            f'{dec}<div class="nt">{cat}</div></div>')


def html_cm(gold, kol, model):
    cm = confusion_matrix(gold["aktual"], gold[kol], labels=URUT)
    r, g, b = RGB[model]
    mx = max(int(cm.max()), 1)
    rows = ""
    for i, a in enumerate(URUT):
        sel = ""
        for j in range(3):
            v = int(cm[i][j])
            al = 0.12 + 0.88 * v / mx
            warna_teks = "#fff" if al > 0.55 else "#1a2333"
            sel += (f'<td><div class="cc" style="background:rgba({r},{g},{b},{al:.2f});'
                    f'color:{warna_teks}">{v}</div></td>')
        rows += f'<tr>{sel}<td class="rl">{a.capitalize()}</td></tr>'
    head = "<tr>" + "".join(f"<th>{k.capitalize()}</th>" for k in URUT) + '<th class="rl">AKTUAL</th></tr>'
    legend = (f'<div class="lg"><span style="background:linear-gradient(90deg,'
              f'rgba({r},{g},{b},.15),rgba({r},{g},{b},1))"></span>Semakin pekat, semakin banyak prediksi</div>')
    return f'<div class="pm">→ PREDIKSI MODEL</div><table class="cmt">{head}{rows}</table>{legend}'


# ---------- data ----------
gold = muat_gold()
semua = muat_semua()
tgl_min, tgl_maks = semua["tanggal"].min(), semua["tanggal"].max()
daftar_versi = sorted(semua["versi_grup"].unique(), key=urut_versi)

# Nilai filter dibaca dari session_state. Widget-nya ada di kartu "Distribusi sentimen"
# (paling bawah), tetapi semua kartu di atasnya ikut berubah karena memakai nilai ini.
tgl_awal = st.session_state.get("f_awal") or tgl_min
tgl_akhir = st.session_state.get("f_akhir") or tgl_maks
versi_pilih = st.session_state.get("f_versi") or []
rentang_ok = tgl_awal <= tgl_akhir


def terapkan(df):
    if not rentang_ok:
        return df.iloc[0:0]
    d = df[(df["tanggal"] >= tgl_awal) & (df["tanggal"] <= tgl_akhir)]
    if versi_pilih:
        d = d[d["versi_grup"].isin(versi_pilih)]
    return d


gold_f = terapkan(gold)
semua_f = terapkan(semua)
ada_gold = len(gold_f) > 0
kelas_ada = [k for k in URUT if (gold_f["aktual"] == k).any()]
kelas_hilang = [k for k in URUT if k not in kelas_ada]

perf_global = pd.DataFrame({"Naive Bayes": hitung(gold, "pred_nb"),
                            "Random Forest": hitung(gold, "pred_rf")}).T
terbaik_global = perf_global["F1-Score"].idxmax()   # dipakai grafik distribusi bila filter tidak memberi kesimpulan

if ada_gold:
    perf = pd.DataFrame({"Naive Bayes": hitung(gold_f, "pred_nb", kelas_ada),
                         "Random Forest": hitung(gold_f, "pred_rf", kelas_ada)}).T
else:
    perf = perf_global
selisih_f1 = abs(perf.loc["Naive Bayes", "F1-Score"] - perf.loc["Random Forest", "F1-Score"])
cukup_data = len(gold_f) >= N_MIN
seri = selisih_f1 < 1e-9
# metode terbaik hanya dinyatakan bila datanya cukup dan F1 kedua model tidak sama
terbaik = perf["F1-Score"].idxmax() if (ada_gold and cukup_data and not seri) else None
model_dist = terbaik or terbaik_global
lainnya = [m for m in MODEL if m != terbaik][0] if terbaik else None
kol_terbaik = "pred_rf" if model_dist == "Random Forest" else "pred_nb"

st.markdown(CSS, unsafe_allow_html=True)

# ---------- Header ----------
with st.container(key="hdr"):
    st.markdown('<div class="hd"><div class="logo">blu</div><div>'
                '<div class="hd-t">Dashboard Perbandingan Naive Bayes vs Random Forest</div>'
                '<div class="hd-s">blu by BCA Digital · Analisis sentimen ulasan aplikasi</div>'
                '</div></div>', unsafe_allow_html=True)

# ---------- Ringkasan filter aktif ----------
def tgl_txt(d):
    return d.strftime("%d/%m/%Y")


versi_txt = ", ".join(versi_pilih) if versi_pilih else "semua versi"
st.markdown(
    f'<div class="fi"><b>Filter aktif</b> · Tanggal {tgl_txt(tgl_awal)} – {tgl_txt(tgl_akhir)} · Versi: {versi_txt}'
    f' · Ulasan sesuai filter: <b>{fmt(len(semua_f))}</b> dari {fmt(len(semua))}'
    f' · Gold set sesuai filter: <b>{fmt(len(gold_f))}</b> dari {fmt(len(gold))}</div>',
    unsafe_allow_html=True)
peringatan = []
if not rentang_ok:
    peringatan.append("Tanggal awal tidak boleh setelah tanggal akhir.")
elif not ada_gold:
    peringatan.append("Tidak ada ulasan gold set pada filter ini, sehingga performa model tidak dapat dihitung.")
else:
    if len(gold_f) < N_MIN:
        peringatan.append(f"Gold set pada filter ini hanya {len(gold_f)} ulasan (kurang dari {N_MIN}); "
                          "hasil kurang stabil, tafsirkan dengan hati-hati.")
    if kelas_hilang:
        peringatan.append("Kelas " + ", ".join(kelas_hilang) + " tidak ada pada gold set filter ini, "
                          "sehingga tidak dihitung dalam rata-rata makro.")
for t in peringatan:
    st.markdown(f'<div class="fw">{t}</div>', unsafe_allow_html=True)

# ---------- Metode Terbaik (dinamis) ----------
with st.container(key="card_metode"):
    if not ada_gold:
        st.markdown('<div class="mt-l">METODE TERBAIK</div>'
                    '<div class="mt-txt" style="margin-top:12px">Tidak ada gold set pada filter ini.</div>',
                    unsafe_allow_html=True)
    elif terbaik is None:
        alasan = (f"gold set pada filter ini hanya {len(gold_f)} ulasan (kurang dari {N_MIN}), terlalu sedikit untuk menyimpulkan"
                  if not cukup_data else "kedua metode memperoleh F1-Score yang sama")
        st.markdown(
            '<div class="mt-top"><span class="mt-l">METODE TERBAIK</span>'
            '<span class="mt-b">Berdasarkan F1-Score makro</span></div>'
            '<div class="mt-mid"><div class="mt-sp"></div><div class="mt-name">Belum dapat disimpulkan</div>'
            '<div class="mt-sp"></div></div>'
            f'<div class="mt-txt">Metode terbaik belum dapat ditentukan karena {alasan}.</div>'
            f'<div class="mt-li">• F1-Score makro: Naive Bayes {pct(perf.loc["Naive Bayes", "F1-Score"])}, '
            f'Random Forest {pct(perf.loc["Random Forest", "F1-Score"])}.</div>'
            f'<div class="mt-st">Dinamis · dihitung dari gold set sesuai filter (n = {len(gold_f)} dari {len(gold)})</div>',
            unsafe_allow_html=True)
    else:
        b_all, c_all, p_all, _, _ = mcnemar(gold_f)
        unggul_kelas = {m: [] for m in MODEL}
        for k in URUT:
            uk = mcnemar(gold_f[gold_f["aktual"] == k])[3]
            if uk in unggul_kelas:
                unggul_kelas[uk].append(k)
        acc = {m: perf.loc[m, "Accuracy"] for m in MODEL}
        if p_all >= ALPHA:
            butir1 = (f"Soal ketepatan menebak secara keseluruhan, keduanya hampir sama: Naive Bayes benar {pct(acc['Naive Bayes'])} "
                      f"dan Random Forest {pct(acc['Random Forest'])}. Selisihnya terlalu kecil untuk dianggap sebagai perbedaan nyata (uji McNemar, p = {p_txt(p_all)}).")
        else:
            lebih = "Naive Bayes" if b_all > c_all else "Random Forest"
            butir1 = (f"Soal ketepatan menebak secara keseluruhan, {lebih} lebih tepat: Naive Bayes benar {pct(acc['Naive Bayes'])} "
                      f"dan Random Forest {pct(acc['Random Forest'])} (uji McNemar, p = {p_txt(p_all)}).")
        bagian = [f"{m} lebih baik mengenali ulasan {' dan '.join(v)}" for m, v in unggul_kelas.items() if v]
        butir2 = ("Bedanya ada pada jenis ulasan (uji McNemar per kelas): " + ", sedangkan ".join(bagian) + ".") if bagian else \
                 "Tidak ada perbedaan nyata pada tiap jenis ulasan."
        st.markdown(
            '<div class="mt-top"><span class="mt-l">METODE TERBAIK</span>'
            '<span class="mt-b">Berdasarkan F1-Score makro</span></div>'
            f'<div class="mt-mid"><div class="mt-sp"></div><div class="mt-name">{terbaik}</div>'
            f'<div class="mt-val">{pct(perf.loc[terbaik, "F1-Score"])}<small>F1-Score</small></div></div>'
            f'<div class="mt-txt">{terbaik} lebih direkomendasikan karena paling seimbang dalam mengenali '
            'ketiga jenis ulasan: positif, negatif, dan netral.</div>'
            f'<div class="mt-li">• {butir1}</div><div class="mt-li">• {butir2}</div>'
            '<div class="mt-ar">F1-Score adalah nilai gabungan yang menilai seberapa adil sebuah metode mengenali ketiga jenis ulasan. '
            f'Makin tinggi, makin seimbang: {terbaik} {pct(perf.loc[terbaik, "F1-Score"])}, '
            f'{lainnya} {pct(perf.loc[lainnya, "F1-Score"])}. '
            'Uji McNemar adalah uji statistik untuk memastikan selisih antar metode bukan sekadar kebetulan.</div>'
            f'<div class="mt-st">Dinamis · dihitung dari gold set sesuai filter (n = {len(gold_f)} dari {len(gold)})</div>',
            unsafe_allow_html=True)


# ---------- Perbandingan performa + McNemar (dinamis) ----------
with st.container(key="card_perf"):
    if not ada_gold:
        st.markdown(judul_kartu("Perbandingan performa model", "Tidak ada gold set pada filter ini", "Dinamis"),
                    unsafe_allow_html=True)
    else:
        st.markdown(judul_kartu(
            "Perbandingan performa model",
            f"Dievaluasi pada gold set sesuai filter (n = {len(gold_f)}) · Precision, Recall, F1-Score: macro-average",
            "Dinamis"), unsafe_allow_html=True)
        kiri, kanan = st.columns([3, 2], gap="large")
        kiri.markdown(html_perf(perf, terbaik) + html_mc_kelas(gold_f), unsafe_allow_html=True)
        kanan.markdown(html_mc_total(gold_f), unsafe_allow_html=True)

# ---------- Confusion matrix (dinamis) + Distribusi sentimen (berisi filter) ----------
c1, c2, c3 = st.columns(3, gap="medium")
for kolom, model, kol, kunci in ((c1, "Naive Bayes", "pred_nb", "card_cm_nb"),
                                 (c2, "Random Forest", "pred_rf", "card_cm_rf")):
    with kolom:
        with st.container(key=kunci):
            if not ada_gold:
                st.markdown(judul_kartu("Confusion matrix", f"{model} · tidak ada gold set pada filter ini",
                                        "Dinamis"), unsafe_allow_html=True)
            else:
                st.markdown(judul_kartu("Confusion matrix", f"{model} · n = {len(gold_f)}", "Dinamis",
                                        pill=pct(perf.loc[model, "Accuracy"]),
                                        kls=f"pill-{SINGKAT[model].lower()}"), unsafe_allow_html=True)
                st.markdown(html_cm(gold_f, kol, model), unsafe_allow_html=True)

with c3:
    with st.container(key="card_dist"):
        st.markdown(judul_kartu("Distribusi sentimen",
                                f"Komposisi ulasan (prediksi {model_dist})" if terbaik else
                                f"Komposisi ulasan (prediksi {model_dist}, model terbaik pada seluruh data)",
                                "Dinamis"),
                    unsafe_allow_html=True)
        with st.container(key="flt"):
            f1, f2 = st.columns(2)
            f1.date_input("Dari tanggal", value=tgl_min, min_value=tgl_min, max_value=tgl_maks, key="f_awal")
            f2.date_input("Sampai tanggal", value=tgl_maks, min_value=tgl_min, max_value=tgl_maks, key="f_akhir")
            st.multiselect("Versi aplikasi (major.minor)", daftar_versi,
                           placeholder="Semua versi", key="f_versi")
        if not rentang_ok:
            st.warning("Tanggal awal tidak boleh setelah tanggal akhir.")
        elif semua_f.empty:
            st.warning("Tidak ada ulasan pada filter ini.")
        else:
            total = len(semua_f)
            hit = semua_f[kol_terbaik].value_counts().reindex(URUT, fill_value=0)
            d = pd.DataFrame({"Sentimen": [k.capitalize() for k in URUT],
                              "Jumlah": [int(hit[k]) for k in URUT]})
            d["Teks"] = d["Jumlah"].map(lambda x: pct(x / total))
            st.markdown(f'<div class="dn">n = {fmt(total)} (sesuai filter)</div>',
                        unsafe_allow_html=True)
            dasar = alt.Chart(d).encode(
                x=alt.X("Sentimen:N", sort=list(d["Sentimen"]), title=None,
                        axis=alt.Axis(labelAngle=0, labelFontSize=13, ticks=False, domain=False)),
                y=alt.Y("Jumlah:Q", axis=None,
                        scale=alt.Scale(domain=[0, max(int(d["Jumlah"].max()) * 1.2, 1)])))
            batang = dasar.mark_bar(size=56, cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                color=alt.Color("Sentimen:N", legend=None,
                                scale=alt.Scale(domain=list(WARNA_DIST),
                                                range=list(WARNA_DIST.values()))))
            label = dasar.mark_text(dy=-10, fontSize=14, fontWeight="bold",
                                    color="#1a2333").encode(text="Teks:N")
            grafik = ((batang + label).properties(height=230)
                      .configure(background="#ffffff").configure_view(stroke=None))
            st.altair_chart(grafik, theme=None, width="stretch")
            st.markdown('<div class="dc">' + " · ".join(
                f"{k.capitalize()} {fmt(int(hit[k]))}" for k in URUT) + "</div>",
                unsafe_allow_html=True)
