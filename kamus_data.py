"""Data statis pipeline (kamus dan daftar kata). Isinya sama persis dengan skrip penelitian:
KAMUS/DOMAIN dari t3_normalisasi.py, NEGASI/EXTRA dari t5_stopword.py, PERBAIKI dari t7_final.py,
NEGATOR/JENDELA/NETRAL/TAMBAHAN dari 10_label_kamus.py."""

# ---------- Tahap 3: normalisasi ----------
KAMUS = {
    # negasi -> tidak / belum
    "ga": "tidak", "gak": "tidak", "gk": "tidak", "g": "tidak", "nggak": "tidak",
    "ngga": "tidak", "ngak": "tidak", "nggk": "tidak", "ngk": "tidak", "nga": "tidak",
    "kaga": "tidak", "engga": "tidak", "enggak": "tidak", "ndak": "tidak", "gx": "tidak",
    "tdk": "tidak",
    "blm": "belum", "blum": "belum", "blom": "belum",
    "gabisa": "tidak bisa", "gbisa": "tidak bisa",
    # kata umum
    "yg": "yang", "udah": "sudah", "udh": "sudah", "sdh": "sudah", "sdah": "sudah",
    "sudh": "sudah", "pake": "pakai", "pke": "pakai", "pakek": "pakai",
    "bgt": "sangat", "bngt": "sangat", "bnget": "sangat", "banget": "sangat", "sgt": "sangat",
    "tp": "tapi", "tpi": "tapi", "klo": "kalau", "kl": "kalau", "klw": "kalau",
    "kalo": "kalau", "karna": "karena", "krn": "karena", "sampe": "sampai",
    "lg": "lagi", "trus": "terus", "trs": "terus", "jg": "juga", "jga": "juga",
    "jugaa": "juga", "jd": "jadi", "utk": "untuk", "dgn": "dengan", "dg": "dengan",
    # kata umum lanjutan
    "dlu": "dulu", "dl": "dulu", "gmn": "bagaimana", "gmna": "bagaimana", "gimana": "bagaimana",
    "gimna": "bagaimana", "gtu": "begitu", "gt": "begitu", "bgtu": "begitu", "gitu": "begitu",
    "bener": "benar", "bner": "benar", "bnr": "benar", "beneran": "benar",
    "tetep": "tetap", "ttp": "tetap", "ttep": "tetap",
    "yaa": "ya", "yahh": "ya", "yak": "ya", "yh": "ya",
    "skrg": "sekarang", "skrng": "sekarang", "skr": "sekarang", "skarang": "sekarang",
    "knp": "kenapa", "knpa": "kenapa", "knapa": "kenapa",
    "bs": "bisa", "bsa": "bisa", "bisaa": "bisa",
    "pdhl": "padahal", "pdhal": "padahal", "pdahal": "padahal",
    "hrs": "harus", "hrus": "harus", "lbh": "lebih", "lbih": "lebih",
    "byk": "banyak", "bnyak": "banyak", "bnyk": "banyak", "smua": "semua", "sma": "sama",
    "msh": "masih", "msih": "masih", "smoga": "semoga",
    "plis": "tolong", "tlong": "tolong", "tlg": "tolong",
    "slalu": "selalu", "sllu": "selalu", "kyk": "seperti", "spt": "seperti", "sperti": "seperti",
    "cm": "cuma", "cmn": "cuma", "cma": "cuma",
    "aj": "saja", "ajh": "saja", "ajah": "saja", "ajaa": "saja", "sj": "saja",
    "emg": "memang", "emng": "memang", "jgn": "jangan", "jngn": "jangan",
    "mo": "mau", "mw": "mau", "dr": "dari", "dri": "dari", "dpt": "dapat", "dapet": "dapat",
    "sy": "saya", "sya": "saya", "aq": "saya", "gw": "saya",
    "terimakasih": "terima kasih", "trimakasih": "terima kasih", "makasih": "terima kasih",
    "tks": "terima kasih", "trims": "terima kasih", "trima": "terima kasih",
    "mksh": "terima kasih", "mks": "terima kasih",
    # merek dan sebutan aplikasi
    "blue": "blu", "bluu": "blu",
    "app": "aplikasi", "apps": "aplikasi", "apk": "aplikasi", "apl": "aplikasi",
    # negasi gabungan
    "gaada": "tidak ada", "gada": "tidak ada", "gajelas": "tidak jelas", "gaje": "tidak jelas",
    "gatau": "tidak tahu", "gapernah": "tidak pernah", "gausah": "tidak usah", "gamau": "tidak mau",
    "d": "di",
    # kata kerja bentuk lisan
    "nunggu": "menunggu", "nungguin": "menunggu", "nunggunya": "menunggu",
    "nabung": "menabung", "nyoba": "mencoba", "nanya": "bertanya", "ngasih": "memberi",
    "ngirim": "mengirim", "ngambil": "mengambil", "ngisi": "mengisi", "ngerti": "mengerti",
    "ngulang": "mengulang", "nyimpen": "menyimpan", "simpen": "simpan",
    "ngebantu": "membantu", "nyesel": "menyesal", "nyusahin": "menyusahkan",
    "ngeselin": "menyebalkan", "masukin": "memasukkan", "tambahin": "menambah",
    "benerin": "memperbaiki", "perbaikin": "memperbaiki",
    "dibenerin": "diperbaiki", "diperbaikin": "diperbaiki",
    "mengunakan": "menggunakan", "dipake": "dipakai", "kepake": "terpakai",
    # slang positif
    "mantab": "mantap", "mantul": "mantap", "mantep": "mantap", "mantapp": "mantap",
    "mantaap": "mantap", "mantabs": "mantap", "mantaf": "mantap", "mantaps": "mantap",
    "mntap": "mantap", "mantull": "mantap", "markotop": "mantap", "jos": "mantap",
    "joss": "mantap", "sipp": "mantap", "siip": "mantap",
    "cepet": "cepat", "satset": "cepat", "gercep": "cepat",
    "kerenn": "keren", "kren": "keren", "kereen": "keren",
    "bgus": "bagus", "baguss": "bagus", "bgs": "bagus",
    "okee": "oke", "okey": "oke", "okeh": "oke", "okay": "oke", "asik": "asyik",
    # sisa kata umum
    "aja": "saja", "ok": "oke", "pd": "pada", "td": "tadi", "tuh": "itu", "tu": "itu", "ni": "ini",
    "tgl": "tanggal", "bln": "bulan", "org": "orang", "orng": "orang", "hr": "hari",
    "males": "malas", "kemaren": "kemarin", "kmrn": "kemarin", "malem": "malam", "temen": "teman",
    "mayan": "lumayan", "stelah": "setelah", "stlh": "setelah",
    "lgsg": "langsung", "lngsung": "langsung", "bru": "baru", "bkin": "bikin", "msuk": "masuk",
    "mulu": "melulu", "samsek": "sama sekali", "sebenernya": "sebenarnya",
    "lgi": "lagi", "jdi": "jadi", "teruss": "terus", "ud": "sudah", "dh": "sudah",
    "sihh": "sih", "sii": "sih", "donk": "dong", "dongg": "dong",
    "tsb": "tersebut", "gabakal": "tidak akan",
    # salah ketik / ejaan istilah
    "pasword": "password", "verif": "verifikasi", "verivikasi": "verifikasi",
    "vermuk": "verifikasi wajah", "tranfer": "transfer", "trf": "transfer", "trx": "transaksi",
    "aktifasi": "aktivasi", "nomer": "nomor", "nmr": "nomor", "komplen": "komplain",
    "telp": "telepon", "telfon": "telepon", "tlp": "telepon", "selfi": "selfie",
    "install": "instal", "uninstall": "uninstal", "unistall": "uninstal", "unistal": "uninstal",
    "loding": "loading", "erorr": "error",
    "rekomen": "rekomendasi", "rekomend": "rekomendasi", "rekomended": "rekomendasi",
    "recomended": "rekomendasi", "recomend": "rekomendasi",
}

# istilah teknis yang boleh kehilangan akhiran -nya (loadingnya -> loading)
DOMAIN = {"loading", "fitur", "admin", "promo", "server", "notif", "cashback", "reward",
          "atm", "hp", "login", "account", "bug", "update", "service", "respon",
          "bca", "cs", "blu", "aplikasi"}
KAMUS["adminya"] = "admin"

# ---------- Tahap 5: stopword ----------
NEGASI = {"tidak", "tak", "bukan", "belum", "tanpa"}   # dipertahankan untuk pelabelan
EXTRA = {"blu", "bca", "aplikasi", "apk", "app", "nya", "yg", "utk", "dgn",
         "sih", "deh", "dong", "kok", "loh", "lho", "nih", "kan", "yah", "toh"}

# ---------- Tahap 7: pengecualian stemming ----------
# Sastrawi mengubah "perbaiki" menjadi "baik" (kata positif di InSet), padahal
# di ulasan hampir selalu berarti keluhan, jadi bentuk ini dikecualikan dari stemming
PERBAIKI = {"perbaiki", "diperbaiki", "memperbaiki", "perbaikan", "perbaikannya"}

# ---------- Pelabelan berbasis kamus ----------
NEGATOR = {"tidak", "tak", "bukan", "belum", "tanpa"}   # tidak dihitung sebagai skor, dipakai untuk membalik
JENDELA = 2                                              # pembalikan berlaku pada 2 token setelah negator

# kata umum / istilah domain tanpa muatan sentimen pada ulasan aplikasi -> bobot 0
NETRAL = {"mau", "transaksi", "tunai", "tarik", "buka", "masuk", "sama", "apa",
          "pas", "sering", "cuma", "semua"}

# kamus tambahan: kata di luar InSet, kata asing, koreksi "lambat", dan kata ambigu yang jelas arahnya
TAMBAHAN = {
    "bagus": 3, "aman": 3, "mudah": 3, "membantu": 3, "bantu": 3,
    "baik": 3, "suka": 3, "cepat": 3, "lancar": 4, "enak": 3, "gampang": 3,
    "memudahkan": 3, "jelas": 2, "aneh": -3,
    "bisa": 2, "keren": 3, "rekomendasi": 3, "recommended": 3, "good": 3, "nice": 3,
    "best": 4, "simple": 2, "easy": 2, "fast": 3, "smooth": 3, "love": 3, "perfect": 4,
    "worth": 2, "friendly": 2,
    "kecewa": -4, "sulit": -3, "kendala": -3, "bug": -3, "ngebug": -3, "lambat": -3,
    "ngelag": -3, "lag": -3, "crash": -4, "stuck": -3, "slow": -3, "sluggish": -3,
    "gangguan": -3, "nyangkut": -3, "bgst": -4, "najis": -4, "sampah": -4,
}
