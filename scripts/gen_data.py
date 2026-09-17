#!/usr/bin/env python
"""Synthetic, instruction-conditional Turkish PII masking data generator.

Produces JSONL rows: {instruction, input, output, kategori, ozellikler, tags, mask_tags}
Targets the weak slices of pii-guard-turkish-270m: uppercase, long text, spelled-out
numbers, whitelist/blacklist/out-of-scope policies, multi-person, [YAS]/[CINSIYET]/[UYRUK].

  python scripts/gen_data.py --n 300000 --out data/train.jsonl --val_out data/val.jsonl --val 4000
"""
import argparse, json, random, re, sys
from collections import Counter

R = random.Random(0)

# ----------------------------------------------------------------------------- Turkish helpers
_UP = str.maketrans("abcçdefgğhıijklmnoöprsştuüvyzqwx", "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZQWX")
def tr_upper(s): return s.translate(_UP)

DIGIT_W = ["sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
TENS_W = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]

def pair_word(n):
    if n < 10: return "sıfır " + DIGIT_W[n]
    return TENS_W[n // 10] + ("" if n % 10 == 0 else " " + DIGIT_W[n % 10])

def spell(numstr, mode=None):
    digits = re.sub(r"\D", "", numstr)
    mode = mode or R.choice(["digit", "digit", "pair"])
    if mode == "digit":
        return " ".join(DIGIT_W[int(c)] for c in digits)
    out = []
    i = 0
    while i < len(digits):
        chunk = digits[i:i + 2]
        out.append(pair_word(int(chunk)) if len(chunk) == 2 else DIGIT_W[int(chunk)])
        i += 2
    return " ".join(out)

VOW = "aeıioöuü"
BACK = set("aıou")
ROUND = set("oöuü")
VOICELESS = set("çfhkpsşt")

def _pron_tail(value):
    """Return the word that determines suffix harmony (digits are read as words)."""
    v = value.rstrip(" .*!#+-_")
    if not v: return "a"
    c = v[-1].lower()
    if c.isdigit(): return DIGIT_W[int(c)]
    return v.lower()

def suffix(value, case):
    w = _pron_tail(value)
    lv = next((ch for ch in reversed(w) if ch in VOW), "a")
    back, rnd = lv in BACK, lv in ROUND
    ends_vowel = w[-1] in VOW
    voiceless = w[-1] in VOICELESS
    two = "a" if back else "e"
    four = ("u" if back else "ü") if rnd else ("ı" if back else "i")
    if case == "GEN": return ("n" if ends_vowel else "") + four + "n"
    if case == "DAT": return ("y" if ends_vowel else "") + two
    if case == "ACC": return ("y" if ends_vowel else "") + four
    if case == "LOC": return ("t" if voiceless else "d") + two
    if case == "ABL": return ("t" if voiceless else "d") + two + "n"
    if case == "INS": return ("y" if ends_vowel else "") + "l" + two   # 'la / 'yla
    raise ValueError(case)

# ----------------------------------------------------------------------------- lexicons
FIRST = """Ahmet Mehmet Mustafa Ali Hüseyin Hasan İbrahim İsmail Osman Yusuf Murat Ömer Emre Burak Serkan Tolga Volkan Kerem
Eren Arda Can Cem Deniz Ege Efe Kaan Onur Ozan Selim Sinan Umut Yiğit Berk Barış Furkan Gökhan Halil Kadir Levent Mert Orhan
Ayşe Fatma Emine Hatice Zeynep Elif Meryem Merve Esra Büşra Zehra Sema Selin Ceren Damla Derya Ebru Eda Ece Gamze Gizem Hande
Nur Nazlı Pınar Seda Sevgi Sibel Tuğba Yasemin Aylin Bahar Beyza Cansu Dilek Duygu Gül Gülşah İrem Kübra Melis Nihan Özge Sude
Aslı Begüm Buse Şeyma Tuba Yeliz Zeliha Rabia Sultan Hülya Nesrin Songül Nurten Gülten Filiz Ayla Semra Neslihan Tülay Sevil
Yasin Yunus Enes Bilal Salih Recep Ramazan Cengiz Erkan Erdem Alper Adem Abdullah Metin Nihat Necati Nuri Sabri Şükrü Tahir
Aleyna Azra Defne Eylül Lina Mila Nehir Öykü Ada Asel Elif Ecrin Miray Zümra Alara Beren Dilara Yağmur İlayda Ilgın Naz Nisa
Alperen Ayaz Bora Çınar Doruk Emir Kuzey Mirac Poyraz Rüzgar Toprak Yağız Atlas Aras Demir Kayra Ömer Tuna Utku Batuhan
Temime Rengül Amre Birsan Zeynelabidin Hıdır Mihriye Beyzade Efser Yoruç Sevsevil Şama Gürarda Orçin Yosma Almus İsra Arıel""".split()
LAST = """Yılmaz Kaya Demir Şahin Çelik Yıldız Yıldırım Öztürk Aydın Özdemir Arslan Doğan Kılıç Aslan Çetin Kara Koç Kurt Özkan Şimşek
Polat Korkmaz Çakır Erdoğan Güneş Akın Turan Aksoy Yavuz Bulut Tekin Uçar Dinç Karaca Manço Tarhan Akgündüz Zengin Gülen Yaman
Hayrioğlu Çamurcuoğlu İhsanoğlu Eroğlu Akçay Bozkurt Sarı Ateş Öz Işık Keskin Taş Erdem Kaplan Avcı Aktaş Duman Gündüz Ergin
Çakar Yayak Ülker Koru Ekin Ergül Toprak Kocaman Sezer Kalkan Bayram Ünal Acar Ateşoğlu Altın Karadağ Özer Mutlu Sağlam Yalçın
Kutlu Tunç Alkan Karahan Tanrıverdi Baştürk Soylu Erten Kırmızı Balcı Koçak Uysal Güler Şen Özçelik Bilgin Aygün Durdu Kor""".split()
CITIES = """İstanbul Ankara İzmir Bursa Antalya Adana Konya Gaziantep Mersin Kayseri Eskişehir Diyarbakır Samsun Denizli Şanlıurfa
Malatya Trabzon Erzurum Van Sakarya Manisa Balıkesir Kocaeli Hatay Tekirdağ Aydın Muğla Ordu Sivas Kahramanmaraş Elazığ Rize
Edirne Çanakkale Isparta Afyon Kütahya Zonguldak Bolu Düzce Yalova Kırklareli Amasya Tokat Çorum Giresun Batman Mardin Muş""".split()
DISTRICTS = """Kadıköy Beşiktaş Üsküdar Şişli Maltepe Kartal Pendik Ümraniye Ataşehir Bakırköy Çankaya Keçiören Yenimahalle Mamak
Konak Karşıyaka Bornova Buca Nilüfer Osmangazi Yıldırım Muratpaşa Kepez Seyhan Çukurova Selçuklu Meram Şahinbey Şehitkamil
Yenişehir Melikgazi Odunpazarı Tepebaşı Pamukkale Merkezefendi Atakum İlkadım Ortahisar Akdeniz Toroslar Efeler Menteşe""".split()
MAH = "Esentepe Gazi Fatih Kültür Yeni Cumhuriyet Karşıyaka Bahçelievler Atatürk Yenimahalle Çamlık Güzelyalı Bağlar Yeşiltepe Zafer Barbaros Hürriyet Kızılay Mimar Sinan".split()
STREET = "Manolya Zambak Ihlamur Çınar Menekşe Lale Akasya Gül Papatya Orkide Nergis Leylak Yasemin Fulya Kardelen Sümbül Çiğdem Mimoza Begonya Sardunya".split()
STREET_T = ["Sokak", "Sok.", "Cad.", "Caddesi", "Bulvarı", "Bulv."]
SECTORS = "Lojistik Bilişim Medya Kimya Turizm İnşaat Tekstil Gıda Enerji Otomotiv Sigorta Yazılım Mobilya Tarım Danışmanlık Plastik Metal Ambalaj Elektronik Ticaret".split()
NATION = "Türk T.C. Alman Suriye Suriyeli Afgan İran İranlı Irak Iraklı Gürcistan Gürcü Ukrayna Ukraynalı Rus Rusya Azeri Azerbaycan Kazakistan Kazak Özbek Özbekistan Türkmen Türkmenistan Kırgız Fransız Fransa İngiliz İngiltere Amerikan ABD Hollanda Hollandalı Bulgar Bulgaristan Yunan Yunanistan Mısır Mısırlı Pakistan Pakistanlı Çin Çinli Hint Hindistan Arnavut Arnavutluk Bosna Bosnalı Kanada Kanadalı".split()
SAGLIK = ["penisilin alerjisi", "hipertansiyon tedavisi", "çölyak hastalığı", "fıstık alerjisi", "tiroit ilacı kullanımı", "kalp yetmezliği takibi",
          "insülin bağımlı diyabet", "astım tanısı", "epilepsi tedavisi", "migren teşhisi", "kronik böbrek yetmezliği", "depresyon tedavisi",
          "bel fıtığı ameliyatı", "kemoterapi süreci", "hepatit B taşıyıcılığı", "demir eksikliği anemisi", "kolesterol ilacı kullanımı", "romatoid artrit",
          "gluten intoleransı", "laktoz intoleransı", "psikiyatrik takip", "kalp pili takılı", "koah tanısı", "sedef hastalığı", "uyku apnesi", "hamilelik takibi",
          "diş implant tedavisi", "göz tansiyonu", "obezite cerrahisi", "alzheimer başlangıcı"]
DIN = ["Müslüman", "Hristiyan", "Musevi", "Agnostik", "Deist", "Ateist", "Alevi", "Sünni", "Katolik", "Ortodoks", "Protestan", "Budist", "Yezidi", "Hanefi", "Şafii", "Caferi", "Bahai", "inançsız"]
ETNIK = ["Laz", "Kürt", "Arnavut", "Süryani", "Rum", "Ermeni", "Çerkes", "Boşnak", "Arap", "Gürcü", "Roman", "Zaza", "Tatar", "Pomak", "Azeri", "Türkmen", "Yörük", "Abhaz", "Çeçen", "Hemşin"]
SENDIKA = ["Koop-İş üyesi", "Öz Finans-İş üyesi", "sendika üyeliği yok", "Basisen üyesi", "Tez-Koop-İş üyesi", "Banka Çalışanları Sendikası üyesi", "Bank-Sen üyesi",
           "Türk-İş üyesi", "DİSK üyesi", "Hak-İş üyesi", "Petrol-İş üyesi", "Türk Metal üyesi", "Eğitim-Sen üyesi", "Birleşik Metal-İş üyesi", "sendikasız", "Öz İplik-İş üyesi", "Tekgıda-İş üyesi"]
BIYO = ["yüz tanıma verisi", "iris taraması", "retina taraması", "ses imzası kaydı", "el geometrisi verisi", "parmak izi şablonu", "damar izi kaydı", "parmak izi", "yüz taraması", "avuç içi izi", "yürüyüş analizi verisi", "ses biyometrisi"]
CEZA = ["icra takibi kaydı", "kesinleşmiş hükmü yok", "kabahat kaydı mevcut", "adli sicil kaydı temiz", "adli para cezası kaydı", "ertelenmiş hapis cezası", "sabıka kaydı yok",
        "hükmün açıklanmasının geri bırakılması", "trafik cezası kaydı", "denetimli serbestlik", "kesinleşmiş sabıka kaydı", "arşiv kaydı mevcut", "dolandırıcılık soruşturması", "beraat kararı"]
ENGEL = ["%40 engelli raporu", "kronik hastalık raporu", "%60 görme engelli", "işitme engelli raporu var", "%50 engelli raporu mevcut", "engel durumu yok", "%90 ağır engelli",
         "ortopedik engelli raporu", "%70 engelli raporu", "zihinsel engelli raporu", "süreğen hastalık raporu", "%30 işitme kaybı raporu", "tekerlekli sandalye kullanıyor", "engelli raporu yenilenecek"]
AILE = ["oğlu üniversitede", "tek çocuklu", "boşanmış, velayet kendisinde", "babası emekli", "eşi vefat etmiş", "eşi ev hanımı", "iki çocuğu var", "eşinden ayrı yaşıyor",
        "üç çocuk annesi", "evli, çocuksuz", "bekar", "annesi bakıma muhtaç", "kardeşi yurtdışında", "eşi aynı bankada çalışıyor", "dul", "boşanma sürecinde", "ikiz çocukları var", "evlat edinilmiş çocuğu var"]
MONTHS = "Ocak Şubat Mart Nisan Mayıs Haziran Temmuz Ağustos Eylül Ekim Kasım Aralık".split()
ALNUM = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
PW_WORDS = "Kar Ege Fener Cagla Ruzgar Marmara Lale Bulut Safak Yaz Deniz Gunes Mavi Yildiz Ankara Izmir Aslan Kartal Toros Palmiye Zeytin Baris Umut".split()
BANKS_MAIL = ["gmail.com", "hotmail.com", "icloud.com", "mynet.com", "yandex.com", "outlook.com", "yahoo.com", "protonmail.com"]

def rd(n): return "".join(R.choice("0123456789") for _ in range(n))
def rnz(n): return R.choice("123456789") + rd(n - 1)
def ral(n, pool=ALNUM): return "".join(R.choice(pool) for _ in range(n))

def person():
    n = R.random()
    if n < 0.72: return f"{R.choice(FIRST)} {R.choice(LAST)}"
    if n < 0.85: return f"{R.choice(FIRST)} {R.choice(FIRST)} {R.choice(LAST)}"
    if n < 0.93: return R.choice(FIRST)
    return f"{R.choice(FIRST)} {R.choice(LAST)}-{R.choice(LAST)}"

def tckn():
    d = [int(c) for c in rnz(9)]
    d10 = ((d[0] + d[2] + d[4] + d[6] + d[8]) * 7 - (d[1] + d[3] + d[5] + d[7])) % 10
    d11 = (sum(d) + d10) % 10
    s = "".join(map(str, d)) + str(d10) + str(d11)
    f = R.random()
    if f < 0.85: return s
    if f < 0.93: return f"{s[:3]}-{s[3:6]}-{s[6:9]}-{s[9:]}"
    return f"{s[:3]} {s[3:6]} {s[6:9]} {s[9:]}"

def tel():
    a, b, c, d = R.choice(["532", "533", "535", "536", "537", "538", "539", "541", "542", "543", "544", "545", "505", "506", "507", "551", "552", "553", "554", "555", "212", "216", "312", "232", "224", "242"]), rd(3), rd(2), rd(2)
    return R.choice([f"0{a} {b} {c} {d}", f"0{a}{b}{c}{d}", f"+90 {a} {b} {c} {d}", f"({a}) {b} {c} {d}", f"0 {a} {b} {c} {d}", f"+90{a}{b}{c}{d}", f"{a} {b} {c}{d}", f"0{a}-{b}-{c}-{d}"])

def luhn16():
    base = [int(c) for c in R.choice(["4", "5", "9792"]) + rd(11)][:15]
    while len(base) < 15: base.append(R.randrange(10))
    s = 0
    for i, x in enumerate(reversed(base)):
        x = x * 2 if i % 2 == 0 else x
        s += x - 9 if x > 9 else x
    return "".join(map(str, base)) + str((10 - s % 10) % 10)

def kart():
    c = luhn16(); f = R.random()
    if f < 0.55: return c
    if f < 0.9: return " ".join(c[i:i + 4] for i in range(0, 16, 4))
    return "-".join(c[i:i + 4] for i in range(0, 16, 4))

def iban():
    d = rd(24); f = R.random()
    if f < 0.4: return "TR" + " ".join(d[i:i + 4] for i in range(0, 24, 4))
    if f < 0.6: return "TR" + d
    if f < 0.8: return "TR" + d[:2] + " " + " ".join(d[2:][i:i + 4] for i in range(0, 22, 4))
    return "TR" + "-".join(d[i:i + 4] for i in range(0, 24, 4))

def hesap():
    f = R.random()
    if f < 0.4: return rd(R.choice([10, 11, 12]))
    if f < 0.7: return f"{rd(4)}-{rd(7)}-{rd(3)}"
    if f < 0.85: return f"{rd(4)} {rd(6)}"
    return f"{rd(3)}-{rd(8)}"

def kart_skt():
    m = f"{R.randint(1, 12):02d}"; y = R.randint(2026, 2034)
    return R.choice([f"{m}/{y % 100}", f"{m}/{y}", f"{m}.{y}", f"{m}-{y % 100}"])

def maas():
    v = R.randint(25, 250) * 1000 + R.choice([0, 500])
    s = f"{v:,}".replace(",", ".")
    return R.choice([f"{s} TL", f"aylık {s} TL", f"{v // 1000} bin TL", f"brüt {s} TL", f"net {s} TL", f"{s} TL maaş", f"{s}₺", f"{v // 1000}.000 TL brüt"])

def musteri_no():
    f = R.random()
    if f < 0.45: return "MU" + rd(7)
    if f < 0.8: return rd(8)
    if f < 0.9: return "M-" + rd(6)
    return "CRM" + rd(6)

def police():
    return R.choice([f"POL-{rd(7)}", f"{rd(4)}/{rd(6)}", f"P{rd(9)}", f"PL-{rd(4)}-{rd(4)}", f"{rd(10)}"])

def sozlesme():
    return R.choice([f"{R.randint(2019, 2027)}/{rd(4)}-{R.choice('ABCDE')}", f"{rd(4)}-{rd(5)}", f"SZL-{rd(6)}", f"{rd(4)}/{rd(3)}-{rd(2)}", f"K{rd(8)}"])

def kripto():
    f = R.random()
    if f < 0.45: return "0x" + ral(40, "0123456789abcdef")
    if f < 0.75: return "bc1q" + ral(30, "023456789acdefghjklmnpqrstuvwxyz")
    return "T" + ral(33, "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz123456789")

_DEASC = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
def ascii_name(s): return s.translate(_DEASC).lower()

def email():
    f, l = ascii_name(R.choice(FIRST)), ascii_name(R.choice(LAST))
    user = R.choice([f"{f}.{l}", f"{f}_{l}", f"{f}{l}{R.randint(1, 99)}", f"{f}{R.randint(1960, 2005)}", f"{f[0]}{l}", f"{l}.{f}", f"{f}{l[:3]}{rd(2)}"])
    return f"{user}@{R.choice(BANKS_MAIL)}"

def adres():
    s = f"{R.choice(MAH)} Mah. {R.choice(STREET)} {R.choice(STREET_T)} No:{R.randint(1, 250)}"
    if R.random() < 0.4: s += f" D:{R.randint(1, 20)}"
    if R.random() < 0.4: s += f" Kat:{R.randint(1, 12)}"
    if R.random() < 0.15: s += f" Daire {R.randint(1, 30)}"
    return f"{s} {R.choice(DISTRICTS)}/{R.choice(CITIES)}"

def konum():
    return f"{R.uniform(36, 42):.4f}, {R.uniform(26, 45):.4f}".replace(".0000", ".0")

def kan(): return R.choice(["A", "B", "AB", "0", "0"]) + R.choice([" Rh pozitif", " Rh negatif", " Rh+", " Rh-", " RH pozitif", " rh negatif"])

def sifre():
    return R.choice(PW_WORDS) + R.choice([".", "_", "-", ""]) + rd(R.choice([3, 4, 4])) + R.choice([".", "!", "*", "#", "+", "?", "", ""])

def pin(): return rd(R.choice([4, 4, 4, 6]))
def kullanici():
    f, l = ascii_name(R.choice(FIRST)), ascii_name(R.choice(LAST))
    return R.choice([f"{f}_{l[:4]}{rd(2)}", f"{f}{l[:3]}{rd(2)}", f"{f[0]}{l}{rd(2)}", f"{f}{l}", f"{l}{rd(4)}", f"{f}.{l}", f"{f}{rd(4)}"])
def ip(): return ".".join(str(R.randint(1, 254)) for _ in range(4))
def mac():
    p = [ral(2, "0123456789ABCDEF") for _ in range(6)]
    return R.choice([":", ":", ":", "-"]).join(p) if R.random() < 0.85 else ":".join(p).lower()
def imei(): return rd(15)
def cihaz(): return R.choice([f"{ral(4)}-{ral(4)}-{ral(4)}", f"DEV-{ral(8)}", f"{ral(6)}-{ral(6)}", ral(12)])
def plaka():
    il = f"{R.randint(1, 81):02d}"; let = ral(R.choice([1, 2, 3]), "ABCDEFGHJKLMNPRSTUVYZ"); num = rd(R.choice([2, 3, 4]) if len(let) == 3 else R.choice([3, 4, 5]))
    return R.choice([f"{il} {let} {num}", f"{il} {let} {num}", f"{il}{let}{num}", f"{il} {let}{num}"])
def sasi(): return ral(17, "ABCDEFGHJKLMNPRSTUVWXYZ0123456789")
def motor(): return R.choice([f"{R.choice('MNOKR')}{rd(2)}{R.choice('BNRA')}{rd(6)}", f"{ral(3)}{rd(7)}", f"ENG{rd(8)}"])
def ruhsat(): return R.choice([f"{ral(2, 'ABCDEFGHJKLMNPRSTUVYZ')}-{rd(6)}", f"{ral(2, 'ABCDEFGHJKLMNPRSTUVYZ')}{rd(6)}", f"{rd(2)}-{ral(2, 'ABCDEFGHJKLMNPRSTUVYZ')}-{rd(5)}"])
def sicil(): return R.choice([f"SC{rd(4)}", rd(4), rd(5), f"P-{rd(5)}", f"PRS{rd(4)}"])
def isyeri():
    f = R.random()
    if f < 0.4: return f"{R.choice(LAST)} {R.choice(SECTORS)} {R.choice(['A.Ş.', 'AŞ', 'Ltd. Şti.', 'Ltd.', 'San. Tic. A.Ş.'])}"
    if f < 0.55: return f"{R.choice(LAST)} {R.choice(LAST)} {R.choice(SECTORS)} {R.choice(['A.Ş.', 'Ltd. Şti.'])}"
    if f < 0.7: return f"{R.choice(CITIES)} Devlet Hastanesi"
    if f < 0.8: return f"{R.choice(CITIES)} Üniversitesi"
    if f < 0.88: return f"{R.choice(CITIES)} Büyükşehir Belediyesi"
    return f"{R.choice(['Atlas', 'Akdeniz', 'Toros', 'Marmara', 'Ege', 'Anadolu', 'Bereket', 'Şafak', 'Kuzey', 'Doğu'])} {R.choice(SECTORS)} {R.choice(['A.Ş.', 'Ltd. Şti.', 'Holding'])}"
def referans():
    f = R.random()
    if f < 0.35: return f"REF-{R.randint(2022, 2027)}-{rd(4)}"
    if f < 0.55: return f"RF{rd(8)}"
    return person()
def yas(): return str(R.randint(18, 89))
def cinsiyet(): return R.choice(["kadın", "erkek", "kadın", "erkek", "Kadın", "Erkek"])
def uyruk(): return R.choice(NATION)
def dogum_tarihi():
    y, m, d = R.randint(1945, 2007), R.randint(1, 12), R.randint(1, 28)
    return R.choice([f"{y}-{m:02d}-{d:02d}", f"{d:02d}.{m:02d}.{y}", f"{d:02d}/{m:02d}/{y}", f"{d} {MONTHS[m - 1]} {y}", f"{d:02d}.{m:02d}.{y}", f"{d}.{m}.{y}"])
def dogum_yeri(): return R.choice(CITIES + DISTRICTS)
def anne_adi(): return R.choice(FIRST)
def baba_adi(): return R.choice(FIRST)
def anne_kizlik(): return R.choice(LAST)
def pasaport(): return R.choice("AUUSTP") + rd(8)
def ehliyet(): return R.choice("ABDEFKM") + rd(7)
def sgk(): d = rd(13); return R.choice([d, f"{d[:4]} {d[4:8]} {d[8:12]} {d[12]}", f"{d[:4]}-{d[4:8]}-{d[8:12]}-{d[12]}"])
def imza(): return R.choice([f"dijital imza özeti {ral(6, '0123456789ABCDEF')}", f"e-imza sertifika seri no {rd(10)}", f"imza hash {ral(8, '0123456789abcdef')}", f"ıslak imza örneği #{rd(4)}", f"imza kodu {ral(6)}"])
def vergi(): return rd(10)
def kredi_notu(): return str(R.randint(1, 1900))
def cvv(): return rd(3)
def saglik(): return R.choice(SAGLIK)
def din(): return R.choice(DIN)
def etnik(): return R.choice(ETNIK)
def sendika(): return R.choice(SENDIKA)
def biyo(): return R.choice(BIYO)
def ceza(): return R.choice(CEZA)
def engel(): return R.choice(ENGEL)
def aile(): return R.choice(AILE)

GEN = {"AD": person, "TCKN": tckn, "DOGUM_TARIHI": dogum_tarihi, "DOGUM_YERI": dogum_yeri, "ANNE_ADI": anne_adi, "ANNE_KIZLIK": anne_kizlik,
       "BABA_ADI": baba_adi, "PASAPORT_NO": pasaport, "EHLIYET_NO": ehliyet, "SGK_NO": sgk, "IMZA": imza, "IBAN": iban, "HESAP_NO": hesap, "KART": kart,
       "KART_SKT": kart_skt, "CVV": cvv, "MAAS": maas, "VERGI_NO": vergi, "MUSTERI_NO": musteri_no, "KREDI_NOTU": kredi_notu, "POLICE_NO": police,
       "SOZLESME_NO": sozlesme, "KRIPTO_CUZDAN": kripto, "TEL": tel, "EMAIL": email, "ADRES": adres, "KONUM": konum, "SAGLIK": saglik, "DIN": din,
       "ETNIK_KOKEN": etnik, "SENDIKA": sendika, "BIYOMETRIK": biyo, "CEZA_KAYDI": ceza, "KAN_GRUBU": kan, "ENGEL_DURUMU": engel, "SIFRE": sifre,
       "PIN": pin, "KULLANICI_ADI": kullanici, "IP_ADRES": ip, "MAC_ADRES": mac, "IMEI": imei, "CIHAZ_ID": cihaz, "PLAKA": plaka, "SASI_NO": sasi,
       "MOTOR_NO": motor, "RUHSAT_NO": ruhsat, "SICIL_NO": sicil, "ISYERI": isyeri, "AILE": aile, "REFERANS": referans, "YAS": yas, "CINSIYET": cinsiyet, "UYRUK": uyruk}
TAGS = list(GEN)
assert len(TAGS) == 53
SPELLABLE = {"TCKN", "TEL", "KART", "IBAN", "MUSTERI_NO", "VERGI_NO", "PIN", "HESAP_NO", "CVV", "SGK_NO", "IMEI"}
DEMOG = {"YAS", "CINSIYET", "UYRUK"}
GROUPS = {
    "ozel": ["SAGLIK", "DIN", "ETNIK_KOKEN", "SENDIKA", "BIYOMETRIK", "CEZA_KAYDI", "KAN_GRUBU", "ENGEL_DURUMU"],
    "finans": ["IBAN", "HESAP_NO", "KART", "KART_SKT", "CVV", "MAAS", "VERGI_NO", "MUSTERI_NO", "KREDI_NOTU", "POLICE_NO", "SOZLESME_NO", "KRIPTO_CUZDAN"],
    "iletisim": ["TEL", "EMAIL", "ADRES", "KONUM"],
    "kimlik": ["AD", "TCKN", "DOGUM_TARIHI", "DOGUM_YERI", "ANNE_ADI", "ANNE_KIZLIK", "BABA_ADI", "PASAPORT_NO", "EHLIYET_NO", "SGK_NO", "IMZA"],
    "arac": ["PLAKA", "SASI_NO", "MOTOR_NO", "RUHSAT_NO"],
    "cihaz": ["IP_ADRES", "MAC_ADRES", "IMEI", "CIHAZ_ID"],
    "kimlik_dogrulama": ["SIFRE", "PIN", "KULLANICI_ADI"],
}
GROUP_NAMES = {"ozel": ["özel nitelikli kişisel veriler", "KVKK madde 6 kapsamındaki hassas veriler", "özel nitelikli veriler"],
               "finans": ["finansal alanlar", "finansal bilgiler", "finansal veriler"],
               "iletisim": ["iletişim bilgileri", "iletişim alanları"],
               "kimlik": ["kimlik bilgileri", "kimlik alanları"],
               "arac": ["araç bilgileri", "araca ait alanlar"],
               "cihaz": ["cihaz ve ağ bilgileri", "cihaz kimlikleri"],
               "kimlik_dogrulama": ["kimlik doğrulama bilgileri", "giriş bilgileri"]}

# Turkish names used inside instructions
NAMES = {
    "AD": ["ad soyad", "isim", "ad ve soyad", "kişi adı", "müşteri adı", "isim soyisim"],
    "TCKN": ["T.C. kimlik numarası", "TC kimlik no", "tc kimlik", "kimlik numarası", "TCKN", "tc no"],
    "DOGUM_TARIHI": ["doğum tarihi", "doğum günü bilgisi"], "DOGUM_YERI": ["doğum yeri"],
    "ANNE_ADI": ["anne adı", "annesinin adı"], "ANNE_KIZLIK": ["anne kızlık soyadı", "kızlık soyadı"], "BABA_ADI": ["baba adı", "babasının adı"],
    "PASAPORT_NO": ["pasaport numarası", "pasaport no"], "EHLIYET_NO": ["ehliyet numarası", "ehliyet no", "sürücü belgesi numarası"],
    "SGK_NO": ["SGK no", "SGK numarası", "sigorta sicil numarası"], "IMZA": ["imza bilgisi", "imza kaydı", "imza verisi"],
    "IBAN": ["IBAN", "IBAN numarası", "iban"], "HESAP_NO": ["hesap numarası", "hesap no"], "KART": ["kart numarası", "kredi kartı numarası", "kart no"],
    "KART_SKT": ["kart son kullanma tarihi", "son kullanma tarihi", "kart skt"], "CVV": ["CVV", "kart güvenlik kodu", "cvv kodu"],
    "MAAS": ["maaş bilgisi", "maaş", "ücret bilgisi"], "VERGI_NO": ["vergi numarası", "vergi no", "VKN"], "MUSTERI_NO": ["müşteri numarası", "müşteri no"],
    "KREDI_NOTU": ["kredi notu", "kredi skoru"], "POLICE_NO": ["poliçe numarası", "poliçe no"], "SOZLESME_NO": ["sözleşme numarası", "sözleşme no"],
    "KRIPTO_CUZDAN": ["kripto cüzdan adresi", "cüzdan adresi", "kripto adresi"], "TEL": ["telefon numarası", "cep telefonu", "telefon", "gsm numarası"],
    "EMAIL": ["e-posta adresi", "e-posta", "mail adresi", "email"], "ADRES": ["adres", "açık adres", "ikamet adresi"], "KONUM": ["konum bilgisi", "konum", "koordinat"],
    "SAGLIK": ["sağlık bilgisi", "sağlık verisi", "hastalık bilgisi"], "DIN": ["din bilgisi", "inanç bilgisi", "din"], "ETNIK_KOKEN": ["etnik köken bilgisi", "etnik köken"],
    "SENDIKA": ["sendika bilgisi", "sendika üyeliği"], "BIYOMETRIK": ["biyometrik veri", "biyometrik kayıt"], "CEZA_KAYDI": ["ceza kaydı", "adli sicil bilgisi", "sabıka kaydı"],
    "KAN_GRUBU": ["kan grubu"], "ENGEL_DURUMU": ["engel durumu", "engellilik bilgisi"], "SIFRE": ["şifre", "parola"], "PIN": ["PIN kodu", "pin", "PIN"],
    "KULLANICI_ADI": ["kullanıcı adı"], "IP_ADRES": ["IP adresi", "ip"], "MAC_ADRES": ["MAC adresi", "mac adresi"], "IMEI": ["IMEI", "IMEI numarası"],
    "CIHAZ_ID": ["cihaz kimliği", "cihaz id", "cihaz ID"], "PLAKA": ["araç plakası", "plaka"], "SASI_NO": ["şasi numarası", "şasi no"], "MOTOR_NO": ["motor numarası", "motor no"],
    "RUHSAT_NO": ["ruhsat numarası", "ruhsat no"], "SICIL_NO": ["sicil numarası", "personel sicil no", "sicil no"], "ISYERI": ["işyeri bilgisi", "çalıştığı kurum", "işyeri"],
    "AILE": ["aile durumu", "aile bilgisi"], "REFERANS": ["referans bilgisi", "referans"], "YAS": ["yaş", "yaş bilgisi"], "CINSIYET": ["cinsiyet", "cinsiyet bilgisi"], "UYRUK": ["uyruk", "uyruk bilgisi", "milliyet"],
}

# ----------------------------------------------------------------------------- sentence templates
T = """
{AD} isimli müşterimiz şubeye gelerek talep oluşturdu
başvuru sahibi {AD} evraklarını tamamladı
{AD:GEN} talebi onay bekliyor
lütfen {AD:DAT} geri dönüş sağlayın
{AD:ABL} gelen dilekçe kayda alındı
görüşme notu {AD} ile öğleden sonra yapıldı
hesap sahibi {AD} tckn {TCKN} bilgileri doğrulandı
{AD} adına {KART} numaralı kart basıldı
vekil {AD} işlemleri {AD} adına yürütecek
{AD} tel {TEL} mail {EMAIL} ile kayıt oluşturuldu
{AD} {YAS} yaşında {CINSIYET} bireysel emeklilik başvurusu yaptı
bilgi güncelleme {AD} doğum tarihi {DOGUM_TARIHI} doğum yeri {DOGUM_YERI}
{AD:ACC} şubeye davet edin
{AD} ve {AD} ortak hesap açtırmak istiyor
kimlik doğrulama tckn {TCKN} ile tamamlandı
{TCKN} tc kimlik numaralı kullanıcı şifre sıfırlama istedi
{TCKN:DAT} ait hesap dökümü hazırlandı
tc {TCKN} vergi no {VERGI_NO} mükellef kaydı açıldı
e-devlet girişi için tc {TCKN} şifre {SIFRE} kullanıldı
{TCKN:GEN} adına açık hesap yok
mükellef vergi numarası {VERGI_NO} beyanname verilmedi
vkn {VERGI_NO} adına fatura düzenlendi
{DOGUM_TARIHI} doğumlu müşteri için yaş sınırı kontrolü yapılsın
doğum tarihi {DOGUM_TARIHI} yanlış girilmiş düzeltilsin
{DOGUM_TARIHI:ABL} önce doğanlar kampanyaya dahil değil
nüfusa kayıtlı olduğu yer {DOGUM_YERI} olarak görünüyor
doğum yeri {DOGUM_YERI} kimlikte farklı yazıyor
güvenlik sorusu anne adı {ANNE_ADI} olarak yanıtlandı
baba adı {BABA_ADI} ile kimlik eşleşmesi yapıldı
kızlık soyadı {ANNE_KIZLIK} sorusu doğru cevaplandı
veli bilgisi anne {ANNE_ADI} baba {BABA_ADI}
anne kızlık soyadı {ANNE_KIZLIK} anne adı {ANNE_ADI} telefon bankacılığı doğrulaması
pasaport numarası {PASAPORT_NO} olan {UYRUK} uyruklu müşteri hesap açtırdı
seyahat sigortası pasaport no {PASAPORT_NO} için düzenlendi
uyruk {UYRUK} ikamet izni kontrol edilecek
{UYRUK} vatandaşı müşteriye çeviri hizmeti verildi
pasaport {PASAPORT_NO} süresi dolmuş yenilenmeli
sürücü belgesi no {EHLIYET_NO} ile araç teslim alındı
ehliyet numarası {EHLIYET_NO} kiralama sözleşmesine işlendi
sgk sicil {SGK_NO} emeklilik hesaplaması istendi
sigorta no {SGK_NO} prim günü sorgusu yapıldı
belge {IMZA} ile onaylandı
sözleşmeye {IMZA} eklendi
maaş hesabı iban {IBAN} olarak tanımlandı
{IBAN} ibanına transfer başarısız oldu
iade ibanı {IBAN} isim {AD}
{IBAN:DAT} eft gönderildi
{HESAP_NO} numaralı hesap dondurulmuş
hesap numarası {HESAP_NO} için ekstre talebi
{KART} numaralı kart yurtdışına kapalı
kart {KART} son kullanma {KART_SKT} güvenlik kodu {CVV} sanal pos hatası
kartın son kullanma tarihi {KART_SKT} geçmiş
cvv {CVV} girilince işlem reddedildi
kart {KART} skt {KART_SKT} ile abonelik ödemesi alınamadı
personelin maaşı {MAAS} olarak bordroya işlendi
gelir belgesi {MAAS} olarak beyan edildi
maaş {MAAS} kredi limiti buna göre hesaplansın
müşteri numarası {MUSTERI_NO} ile çağrı açıldı
{MUSTERI_NO} müşteri nolu hesapta bloke var
findeks notu {KREDI_NOTU} başvuru reddedildi
kredi skoru {KREDI_NOTU} olan müşteriye teklif sunulsun
{POLICE_NO} poliçe numaralı sigorta iptal edildi
poliçe {POLICE_NO} prim borcu var
{SOZLESME_NO} numaralı sözleşme yenilendi
sözleşme numarası {SOZLESME_NO} arşivden çıkarıldı
kripto varlık transferi cüzdan {KRIPTO_CUZDAN} adresine yapıldı
{KRIPTO_CUZDAN} adresine gönderim beklemede
müşteri {TEL} numarasından ulaşılabilir
iletişim numarası {TEL} olarak güncellendi
{TEL:DAT} sms gönderildi
acil durumda {TEL} aranacak
bildirimler {EMAIL} adresine gidiyor
e-posta {EMAIL} sistemde kayıtlı değil
{EMAIL} adresli kullanıcı aboneliği iptal etti
fatura adresi {ADRES} olarak kaydedildi
kargo {ADRES} adresine teslim edildi
{ADRES:DAT} taşındığını bildirdi
ikametgah {ADRES}
cihazın son görüldüğü koordinat {KONUM}
işlem {KONUM} konumundan yapıldı
sağlık beyanı {SAGLIK} sigorta teklifi buna göre çıkacak
raporda {SAGLIK} tanısı yer alıyor
hastalık bilgisi {SAGLIK} ile başvuru yapıldı
inanç bilgisi {DIN} olarak kayıtlı
dini {DIN} olarak beyan etti
etnik kökeni {ETNIK_KOKEN} olarak işaretlenmiş
köken alanında {ETNIK_KOKEN} yazıyor
sendika üyeliği {SENDIKA} aidat kesintisi yapılacak
sendika alanı {SENDIKA} olarak güncellendi
giriş {BIYOMETRIK} ile doğrulandı
{BIYOMETRIK} kaydı yenilenecek
sabıka sorgusu sonucu {CEZA_KAYDI}
adli sicil {CEZA_KAYDI} olarak döndü
kan grubu {KAN_GRUBU} olarak sağlık formuna işlendi
acil kartında {KAN_GRUBU} yazıyor
engellilik bilgisi {ENGEL_DURUMU} indirim uygulanacak
{ENGEL_DURUMU} beyanı ile başvuru yapıldı
geçici şifre {SIFRE} olarak atandı
parola {SIFRE} ile giriş başarısız
yeni şifrem {SIFRE} çalışmıyor
kart pin kodu {PIN} üç kez hatalı girildi
pin {PIN} sıfırlansın
kullanıcı {KULLANICI_ADI} yetkileri kaldırıldı
{KULLANICI_ADI} kullanıcı adıyla giriş yapıldı
{KULLANICI_ADI:GEN} hesabı askıya alındı
giriş ip adresi {IP_ADRES} yurtdışı görünüyor
{IP_ADRES} ip adresinden çok sayıda deneme yapıldı
ip {IP_ADRES} beyaz listeye alınsın
cihaz mac {MAC_ADRES} ağ erişimi engellendi
mac adresi {MAC_ADRES} tanımlı değil
imei numarası {IMEI} kayıtlı cihaz listesinde yok
{IMEI} imei nolu telefon bloke edildi
cihaz kimliği {CIHAZ_ID} güvenilir cihazlara eklendi
mobil uygulama cihaz id {CIHAZ_ID} eşleşmedi
{PLAKA} plakalı araç otoparka giriş yaptı
araç plakası {PLAKA} trafik sigortası yenilendi
plaka {PLAKA} ceza sorgusu yapıldı
şasi numarası {SASI_NO} hasar kaydı var
{SASI_NO} şasi nolu araç rehinli
motor numarası {MOTOR_NO} ruhsatla uyumlu
motor {MOTOR_NO} şasi {SASI_NO} ekspertiz raporu hazır
ruhsat numarası {RUHSAT_NO} yenileme başvurusu alındı
araç ruhsat no {RUHSAT_NO} sisteme eklendi
personel sicil {SICIL_NO} izin talebi onaylandı
sicil numarası {SICIL_NO} bordro hatası bildirdi
işyeri bilgisi {ISYERI} olarak beyan edildi
çalıştığı kurum {ISYERI} maaş promosyonu tanımlandı
{ISYERI} çalışanı {AD} kredi başvurusu yaptı
aile bilgisi {AILE} olarak notlandı
sosyal yardım formu aile durumu {AILE}
referans {REFERANS} ile başvuru yapıldı
referans bilgisi {REFERANS} teyit edildi
{YAS} yaşındaki {CINSIYET} müşteriye emekli maaşı promosyonu verildi
yaş {YAS} cinsiyet {CINSIYET} uyruk {UYRUK} demografik kayıt
müşteri {YAS} yaşında ve {CINSIYET}
cinsiyet alanı {CINSIYET} olarak düzeltildi
{YAS} yaş üstü olduğu için ek belge istendi
kayıt açılışı {AD} tc {TCKN} tel {TEL} adres {ADRES}
kart teslimi {AD} kart {KART} pin {PIN}
hesap açılışı {AD} iban {IBAN} müşteri no {MUSTERI_NO}
log: user={KULLANICI_ADI} ip={IP_ADRES} device={CIHAZ_ID}
ekstre {EMAIL} adresine {AD} adına gönderildi
{AD} {DOGUM_TARIHI} {TCKN} kimlik fotokopisi eklendi
şikayet: {AD} {TEL} kart {KART} çift çekim
{AD} adına kayıtlı {PLAKA} plakalı araç haczedildi

{HESAP_NO:DAT} otomatik ödeme talimatı tanımlansın
{HESAP_NO:GEN} bakiyesi eksiye düştü
{HESAP_NO:ABL} çıkan havale iade edildi
{KART:GEN} limiti artırılsın
{KART:DAT} taksit tanımlanamıyor
{KART:GEN} son dört hanesi ile doğrulama yapıldı
{PLAKA:GEN} otoyol geçişi ücretlendirildi
{PLAKA:DAT} kesilen ceza itiraz edildi
{MUSTERI_NO:DAT} özel kampanya tanımlandı
{MUSTERI_NO:GEN} talebi kapatıldı
{IBAN:DAT} yapılan havale beklemede
{IBAN:GEN} sahibi ile isim uyuşmuyor
{SOZLESME_NO:GEN} süresi doldu
{POLICE_NO:DAT} ait hasar dosyası kapatıldı
{TEL:GEN} üzerine kayıtlı abonelik sorgulandı
{IMEI:GEN} garanti kaydı bulunamadı
{SGK_NO:DAT} ait prim borcu yok
{VERGI_NO:GEN} borç sorgusu yapıldı
{PASAPORT_NO:GEN} geçerlilik tarihi kontrol edildi
{EHLIYET_NO:GEN} ceza puanı sorgulandı
{SICIL_NO:GEN} izin bakiyesi güncellendi
{CIHAZ_ID:GEN} eşleştirmesi kaldırıldı
{IP_ADRES:ABL} yapılan istekler engellendi
{KULLANICI_ADI:DAT} yeni rol atandı
{EMAIL:DAT} ekstre de gönderilsin
{TEL:DAT} bilgi mesajı da gitsin
{AD} {ISYERI} personeli olarak kayıtlı
{AD} {TEL} numarasından da ulaşılabilir
personel {AD} {SICIL_NO} sicil ile {ISYERI} bordrosunda
{AD} {DOGUM_YERI} doğumlu {UYRUK} uyruklu
{AD} {ADRES} adresinde ikamet ediyor
{AD} {EMAIL} {TEL} iletişim kartı güncellendi
ödeme {IBAN} {AD} adına {MAAS} tutarında yapıldı
{KART} {KART_SKT} {CVV} ile online ödeme denendi
{PLAKA} {SASI_NO} {MOTOR_NO} araç kaydı tamamlandı
{IP_ADRES} {MAC_ADRES} {CIHAZ_ID} cihaz parmak izi alındı
{USERFILL} müşteri {AD} tel {TEL} arasın
{USERFILL} {AD:GEN} ekstresi de {EMAIL} adresine gitsin
{USERFILL} {TCKN} için de kimlik doğrulaması yapılsın
sağlık raporunda {SAGLIK} yazıyor ama sigorta primi değişmedi
adli sicil belgesinde {CEZA_KAYDI} ibaresi var ama işe alım tamamlandı
hem {AD} hem {AD} aynı adreste görünüyor
hem {TEL} hem {EMAIL} güncellenecek
{AD} için de {KART} kartı yeniden basılsın
""".strip().split("\n")
TEMPLATES = [t.strip() for t in T if t.strip()]
PH = re.compile(r"\{([A-Z_]+)(?::([A-Z]+))?\}")
EKLI_T = [t for t in TEMPLATES if re.search(r"\{[A-Z_]+:[A-Z]+\}", t)]
PLAIN_T = [t for t in TEMPLATES if t not in EKLI_T]

REC_PREFIX = ["sisteme girilecek kayıt:", "yeni müşteri kaydı:", "bayi ekranından gelen kayıt:", "başvuru detayları:", "kredi başvuru formu:",
              "profil güncellemesi:", "kayıt formu:", "crm notu:", "çağrı merkezi kaydı:", "onboarding formu:", "personel kartı:", "müşteri güncellemesi:",
              "kyc formu:", "şube notu:", "form verisi:", "başvuru:", ""]
REC_LABEL = {
    "AD": ["ilgili kişi", "müşteri", "ad soyad", "isim", "adı"], "TCKN": ["tc kimlik no", "tc si", "tckn", "kimlik no", "tc"],
    "DOGUM_TARIHI": ["doğum tarihi", "d. tarihi"], "DOGUM_YERI": ["doğum yeri"], "ANNE_ADI": ["anne adı"], "ANNE_KIZLIK": ["anne kızlık soyadı", "kızlık soyadı"],
    "BABA_ADI": ["baba adı"], "PASAPORT_NO": ["pasaport no", "pasaport"], "EHLIYET_NO": ["ehliyet no"], "SGK_NO": ["sgk no", "sgk"], "IMZA": ["imza kaydı", "imza"],
    "IBAN": ["ıban", "iban"], "HESAP_NO": ["hesap no"], "KART": ["kart no", "kart"], "KART_SKT": ["skt", "kart skt", "son kullanma"], "CVV": ["cvv", "cvv si", "güvenlik kodu"],
    "MAAS": ["maaşı", "maaş", "ücret"], "VERGI_NO": ["vergi no", "vkn"], "MUSTERI_NO": ["müşteri no"], "KREDI_NOTU": ["kredi notu"], "POLICE_NO": ["poliçe no"],
    "SOZLESME_NO": ["sözleşme no"], "KRIPTO_CUZDAN": ["cüzdan adresi", "kripto cüzdan"], "TEL": ["tel", "telefon", "gsm", "cep"], "EMAIL": ["mail adresi", "e-posta", "mail"],
    "ADRES": ["adresi", "adres"], "KONUM": ["konum", "son konum"], "SAGLIK": ["sağlık notu", "sağlık"], "DIN": ["din hanesi", "din"], "ETNIK_KOKEN": ["köken bilgisi", "etnik köken"],
    "SENDIKA": ["sendika bilgisi", "sendika"], "BIYOMETRIK": ["biyometrik kaydı", "biyometrik"], "CEZA_KAYDI": ["adli kaydı", "adli sicil", "ceza kaydı"],
    "KAN_GRUBU": ["kan grubu"], "ENGEL_DURUMU": ["engel durumu"], "SIFRE": ["şifresi", "şifre", "parola"], "PIN": ["pin", "pin i"], "KULLANICI_ADI": ["kullanıcı adı"],
    "IP_ADRES": ["ip", "ip adresi"], "MAC_ADRES": ["mac", "mac adresi"], "IMEI": ["imei"], "CIHAZ_ID": ["cihaz id", "eşleştirme kodu", "cihaz"], "PLAKA": ["plaka"],
    "SASI_NO": ["şasi no", "şasi"], "MOTOR_NO": ["motor no"], "RUHSAT_NO": ["ruhsat no"], "SICIL_NO": ["sicil no", "sicil"], "ISYERI": ["işyeri", "kurum"],
    "AILE": ["aile durumu"], "REFERANS": ["referansı", "referans"], "YAS": ["yaşı", "yaş"], "CINSIYET": ["cinsiyeti", "cinsiyet"], "UYRUK": ["uyruğu", "uyruk"],
}

# ----------------------------------------------------------------------------- negatives (traps)
def _hh(): return f"{R.randint(8, 21):02d}"
NEG = [
    "sipariş {n6} kargoya verildi", "irsaliye no {n5} onaylandı", "fatura tutarı {n5} TL {date} vadeli", "{date} tarihli toplantı iptal edildi",
    "kampanya {date} tarihinde sona eriyor", "ödeme son günü {date}", "{city} şubesi cumartesi açık", "{city} bölge toplantısı {hh}:00 de",
    "yeni şube {street} Caddesi üzerinde açılacak", "tc kimlik numarası girilmeden devam edilemez", "iban formatı TR ile başlamalı", "kart numarası 16 haneli olmalı",
    "telefon alanına sadece rakam girin", "e-posta formatı geçersiz uyarısı çıkıyor", "adres bilgisi eksik girilmiş", "şifre büyük harf içermeli",
    "pin 4 haneli olmalı", "kullanıcı adı alınmış hatası veriyor", "bakiye {n5} TL", "limit {n5} TL ye yükseltildi", "kur {n2},{n2} seviyesinde kapandı",
    "faiz yüzde {n1},{n2} olarak açıklandı", "komisyon {n3} TL kesildi", "hata kodu ERR-{n4} alınıyor", "işlem no {n8} iptal edildi",
    "sağlık taraması cuma günü yapılacak", "sendika temsilcisi seçimi yarın", "dini bayram öncesi mesai {hh}:00 da bitecek", "engelli erişim rampası yapıldı",
    "kadınlar günü etkinliği düzenlenecek", "erkek personel için forma dağıtıldı", "yaş ortalaması {n2} olarak hesaplandı", "uyruk bilgisi zorunlu değil",
    "araç filosu {n2} araca çıktı", "motor yağı değişimi yapıldı", "şasi kontrolü tamamlandı", "ruhsat yenileme ücreti {n4} TL oldu", "yeni cihazlar {n2} adet geldi",
    "ip aralığı değişti", "mac filtreleme açıldı", "imei kayıt ücreti {n5} TL", "işyeri hekimi perşembe gelecek", "personel yemek ücreti {n3} TL",
    "toplantı odası {n3} rezerve edildi", "kat {n1} deki yazıcı bozuk", "masa no {n2} boşaldı", "kapı kodu değişti", "{proj} projesi bütçesi onaylandı",
    "Atatürk Caddesi şubesi tadilatta", "Barbaros Bulvarı şubesi kapandı", "Cumhuriyet Mahallesi şubesi taşındı", "{city} Havalimanı şubesi 24 saat açık",
    "aylık rapor {date} de yayınlanacak", "stok sayımı {date} tarihinde yapılacak", "sürüm {n1}.{n2} yayına alındı", "sunucu bakımı {hh}:00 - {hh}:30 arası",
    "kredi faizleri yüzde {n2} düştü", "mevduat faizi yüzde {n2} oldu", "pos komisyonu yüzde {n1},{n2}", "havale ücreti {n2} TL",
    "anne adı alanı boş bırakılamaz", "baba adı sorusu kaldırıldı", "kızlık soyadı alanı forma eklendi", "doğum tarihi formatı gg.aa.yyyy olmalı",
    "doğum yeri alanı listeden seçilmeli", "pasaport fotokopisi gerekli", "ehliyet sınıfı B olmalı", "sgk kaydı olmayan başvuramaz", "imza atılmadan işlem yapılamaz",
    "iban paylaşırken dikkatli olun", "hesap numarası yerine iban kullanın", "kart bilgilerinizi kimseyle paylaşmayın", "son kullanma tarihi geçmiş kartlar iptal edildi",
    "cvv kodu istenmez", "maaş günü ayın {n2} si", "vergi levhası güncellendi", "müşteri numarası sms ile gelecek", "kredi notu ücretsiz sorgulanabilir",
    "poliçe şartları değişti", "sözleşme örneği ektedir", "kripto işlemleri geçici olarak durduruldu", "telefon bankacılığı {hh}:00 e kadar açık",
    "e-posta bildirimleri kapatıldı", "adres değişikliği şubeden yapılır", "konum servisleri kapalı", "sağlık sigortası kampanyası başladı", "din bayramında şubeler kapalı",
    "etnik yemek festivali düzenleniyor", "sendika aidatları güncellendi", "biyometrik okuyucu arızalı", "adli tatil {date} de başlıyor", "kan bağışı otobüsü geldi",
    "engelli park alanı genişletildi", "şifre değişikliği zorunlu hale geldi", "pin denemesi {n1} ile sınırlı", "kullanıcı adı büyük küçük harf duyarlı",
    "ip adresi otomatik atanır", "mac adresi etiketi cihazın altında", "imei sorgusu ücretsiz", "cihaz kaydı {n1} dakika sürer", "plaka tanıma sistemi devrede",
    "şasi numarası ruhsatta yazar", "motor gücü {n3} beygir", "ruhsat sahibi değişikliği noterden", "sicil dosyaları dijitale taşındı", "işyeri açılış saati {hh}:00",
    "aile boyu kampanya başladı", "referans mektubu zorunlu değil", "yaş sınırı kaldırıldı", "cinsiyet ayrımı yapılmaz", "uyruk fark etmeksizin hesap açılır",
    "toplam {n3} başvuru alındı", "{n2} personel izinli", "{n4} adet kart basıldı", "günlük işlem sayısı {n5}", "haftalık rapor {n2} sayfa",
    "borsa endeksi {n4} puan", "altın gramı {n4} TL", "dolar {n2},{n2} TL", "euro {n2},{n2} TL", "petrol {n2},{n1} dolar",
    "kasa sayımı {hh}:{mm} de yapılacak", "vardiya {hh}:00 de başlıyor", "servis {hh}:{mm} de kalkıyor", "öğle arası {hh}:00 - {hh}:00",
    "hesap dönemi kapanışı {date}", "bütçe revizyonu {month} ayında", "denetim {month} {n4} de yapılacak", "yıl sonu kapanışı {n4}",
    "proje kodu {proj}-{n3}", "kampanya kodu BAHAR{n4}", "indirim kodu {n2}OFF", "etkinlik kodu EVT-{n5}", "belge no DOC-{n6}",
    "stok kodu {n5} güncellendi", "ürün kodu {n6} pasif", "barkod {n9} okunmuyor", "seri no alanı boş", "parti numarası {n5}",
    "kat mülkiyeti {n2} daire", "arsa {n4} metrekare", "kira {n5} TL", "aidat {n4} TL", "depozito {n5} TL",
]
NEG_SUBJ = ["şube müdürü", "muhasebe ekibi", "it departmanı", "insan kaynakları", "çağrı merkezi", "kredi komitesi", "uyum birimi", "operasyon ekibi",
            "satın alma", "hukuk birimi", "yönetim kurulu", "bölge müdürlüğü", "kart operasyonları", "dijital bankacılık", "risk yönetimi", "iç denetim",
            "bilgi güvenliği", "pazarlama ekibi", "şube personeli", "saha ekibi", "teknik servis", "kurumsal iletişim", "hazine birimi", "kalite ekibi"]
NEG_PRED = ["toplantıyı {hh}:{mm} e aldı", "raporu {n2} sayfa olarak hazırladı", "{n3} adet talebi kapattı", "bütçeyi yüzde {n2} artırdı", "yeni prosedürü yayınladı",
            "eğitimi {date} tarihine erteledi", "sistem bakımını duyurdu", "{n2} personel için izin planladı", "kampanyayı {month} ayına taşıdı", "denetim takvimini paylaştı",
            "{n4} numaralı sürümü test ediyor", "stok sayımını tamamladı", "{n5} TL tutarındaki faturayı onayladı", "haftalık hedefi {n3} olarak belirledi", "yeni pos cihazlarını dağıttı",
            "kasa açığını {n4} TL olarak raporladı", "toplantı odasını {n3} olarak değiştirdi", "sürüm notlarını gönderdi", "hata kaydı ERR-{n4} ü kapattı", "{n2} şubeye tebligat gönderdi",
            "kimlik doğrulama akışını güncelledi", "iban formatı kontrolünü sıkılaştırdı", "kart limit politikasını değiştirdi", "şifre kurallarını yeniledi", "adres doğrulama servisini açtı",
            "maaş promosyonu anlaşmasını imzaladı", "sağlık sigortası teklifini reddetti", "sendika görüşmesini erteledi", "engelli erişim projesini başlattı", "araç filosu ihalesini açtı",
            "cihaz envanterini güncelledi", "ip beyaz listesini yeniledi", "plaka tanıma sistemini devreye aldı", "ruhsat işlemlerini e-devlete taşıdı", "personel sicil arşivini taradı",
            "referans kontrol sürecini kaldırdı", "yaş sınırı politikasını güncelledi", "uyruk alanını opsiyonel yaptı", "doğum günü kutlamasını {hh}:00 a aldı", "kan bağışı etkinliği düzenledi"]
PROJ = ["Fatih", "Anadolu", "Marmara", "Ege", "Karadeniz", "Toros", "Kuzey", "Güneş", "Zafer", "Atlas", "Boğaziçi", "Kapadokya"]

def neg_text():
    t = R.choice(NEG) if R.random() < 0.6 else R.choice(NEG_SUBJ) + " " + R.choice(NEG_PRED)
    def sub(m):
        k = m.group(1)
        if k.startswith("n"): return rnz(int(k[1:])) if int(k[1:]) >= 2 else rd(1)
        if k == "hh": return _hh()
        if k == "mm": return R.choice(["00", "15", "30", "45"])
        if k == "date": y, mo, d = R.randint(2024, 2028), R.randint(1, 12), R.randint(1, 28); return R.choice([f"{d:02d}.{mo:02d}.{y}", f"{y}-{mo:02d}-{d:02d}", f"{d:02d}/{mo:02d}/{y}", f"{d} {MONTHS[mo-1]} {y}"])
        if k == "city": return R.choice(CITIES)
        if k == "street": return R.choice(STREET)
        if k == "month": return R.choice(MONTHS)
        if k == "proj": return R.choice(PROJ)
        return m.group(0)
    return re.sub(r"\{(\w+)\}", sub, t)

# ----------------------------------------------------------------------------- segment builder
FILL_PRE = ["lütfen", "acil", "bilginize", "not:", "tekrar", "bugün", "dün", "ayrıca", "hatırlatma:", "önemli:", "rica ederim", "bir de", "ek olarak", "hala"]
FILL_POST = [" lütfen", " teşekkürler", " acil", " bilginize", " en kısa sürede", "?", " rica ederim", " dönüş bekliyorum", " not düşüldü", " onay bekliyor"]

def fill(template, sozle):
    """template -> list of segments: str or (tag, value)."""
    template = template.replace("{USERFILL}", R.choice(FILL_PRE))
    if R.random() < 0.12: template = R.choice(FILL_PRE) + " " + template
    if R.random() < 0.12: template = template + R.choice(FILL_POST)
    segs, pos = [], 0
    for m in PH.finditer(template):
        segs.append(template[pos:m.start()])
        tag, case = m.group(1), m.group(2)
        val = GEN[tag]()
        if sozle and tag in SPELLABLE and R.random() < 0.75:
            if tag == "IBAN": val = "TR " + spell(val[2:]) if R.random() < 0.8 else "TR" + val[2:]
            else: val = spell(val)
        segs.append((tag, val))
        if case: segs.append("'" + suffix(val, case))
        pos = m.end()
    segs.append(template[pos:])
    return [s for s in segs if s != ""]

def record(sozle, k=None):
    k = k or R.randint(5, 9)
    tags = R.sample(TAGS, k)
    sep = R.choice([", ", ", ", ", ", " | ", "; ", " - ", " / "])
    segs = []
    pre = R.choice(REC_PREFIX)
    if pre: segs.append(pre + " ")
    for i, tag in enumerate(tags):
        val = GEN[tag]()
        if sozle and tag in SPELLABLE and R.random() < 0.6: val = spell(val, "pair" if R.random() < 0.5 else "digit")
        lab = R.choice(REC_LABEL[tag])
        segs.append(("" if i == 0 else sep) + lab + R.choice([" ", " ", " ", ": ", "="]))
        segs.append((tag, val))
    return segs


def build_text(kind, sozle, ekli):
    if kind == "kayit":
        return record(sozle)
    pool = EKLI_T if ekli else PLAIN_T
    if kind == "duz":
        segs = fill(R.choice(pool), sozle)
        if R.random() < 0.12: segs = [R.choice(["not: ", "talep: ", "müşteri notu: ", "log: ", "çağrı özeti: ", "ticket: ", "mesaj: "])] + segs
        if R.random() < 0.15 and isinstance(segs[0], str): segs[0] = segs[0][0].upper().replace("I", "İ") if segs[0][0] == "i" else segs[0][0].upper() + segs[0][1:] if len(segs[0]) > 1 else segs[0]
        return segs
    if kind == "cok_kisi":
        parts = [fill(R.choice(PLAIN_T + EKLI_T), sozle) for _ in range(R.choice([2, 2, 2, 3]))]
        segs = parts[0]
        for p_ in parts[1:]: segs = segs + [R.choice([" ayrıca ", " ayrıca ", "; ayrıca ", ". ayrıca ", " ve ", ", öte yandan ", " bunun yanında "])] + p_
        return segs
    if kind == "uzun":
        n = R.randint(3, 6)
        segs = []
        for i in range(n):
            if R.random() < 0.15: part = record(sozle, R.randint(3, 5))
            else: part = fill(R.choice(pool if ekli and R.random() < 0.5 else TEMPLATES), sozle)
            if i: segs.append(R.choice([". ", ". ", ". ", ". ", "; ", " | "]))
            segs += part
        return segs
    raise ValueError(kind)

def render(segs, mask, caps):
    inp, out = [], []
    for s in segs:
        if isinstance(s, str):
            v = tr_upper(s) if caps else s
            inp.append(v); out.append(v)
        else:
            tag, val = s
            v = tr_upper(val) if caps else val
            inp.append(v); out.append(f"[{tag}]" if tag in mask else v)
    return "".join(inp), "".join(out)

# ----------------------------------------------------------------------------- instructions
FULL = [
    "Metindeki tüm kişisel verileri uygun etiketlerle maskele.",
    "KVKK gereği bütün kişisel ve hassas alanları gizle; tutar, adet gibi kişisel olmayan sayılara dokunma.",
    "Bütün PII alanlarını maskele, metnin geri kalanını aynen koru.",
    "Bu kayıtta kişisel veri niteliği taşıyan her şeyi köşeli parantezli etiketle değiştir.",
    "Bu satırdaki her kişisel bilgiyi maskeleyerek yeniden yaz.",
    "Aşağıdaki mesajda kimliği belirlenebilir kılan ne varsa etiketiyle ört.",
    "Kişisel verileri tespit edip ilgili etiketlerle değiştir; başka değişiklik yapma.",
    "Gizlilik filtresi uygula: tüm kişisel alanlar etikete dönüşsün.",
    "Tüm kişisel verileri maskele.",
    "Metni anonimleştir: her PII değerini uygun etiketle değiştir.",
    "Kişisel veri içeren tüm alanları etiketle, metnin yapısını bozma.",
    "Bu metindeki bütün hassas ve kişisel bilgileri maskeleyerek geri döndür.",
    "Aşağıdaki metni KVKK'ya uygun hale getir: kişisel verileri etiketlerle değiştir.",
    "Loglama öncesi temizlik: kişisel verileri ilgili etiketle örterek metni yeniden yaz.",
    "Metindeki her kişisel veriyi köşeli parantez içinde etiketle maskele; başka hiçbir kelimeyi değiştirme.",
    "Kişiyi tanımlayabilecek tüm bilgileri maskele, kişisel olmayan sayı ve tutarları koru.",
    "Tüm PII değerlerini etikete çevir.",
    "Bu mesajı LLM'e göndermeden önce içindeki kişisel verileri etiketlerle maskele.",
    "Kimlik, finans, iletişim ve özel nitelikli tüm alanları maskele.",
    "Hassas veri filtresi: metindeki kişisel verilerin tamamını etiketle.",
    "Metinde geçen bütün kişisel verileri maskele; PII yoksa metni olduğu gibi döndür.",
    "Tüm kişisel verileri etiketle. Kişisel veri içermiyorsa metne dokunma.",
    "GDPR/KVKK maskeleme: kişisel verileri etiketlerle değiştir, geri kalanı koru.",
    "Her PII alanını uygun etiketle değiştir.",
]
WHITE = [
    "Sadece {X} için maskeleme yap, gerisi olduğu gibi kalsın.", "Bu metinde {X} dışında hiçbir şeyi maskeleme.",
    "Kapsam yalnızca {X}; diğer kişisel veriler bu sefer açık kalacak.", "{X} alanını etiketle, kalan her şeyi aynen bırak.",
    "Yalnız {X} gizlenecek; başka hiçbir şey değişmeyecek.", "Maskeleme talimatı: sadece {X}.", "Bu talepte tek hedef {X}; başka alan etiketlenmeyecek.",
    "Sadece {X} maskele. Geri kalan her şey aynen kalsın.", "Yalnızca {X} etiketle.", "{X} maskelenecek, diğer alanlar açık kalacak.",
    "Sadece şu alanları maskele: {X}.", "Bu metinde yalnızca {X} maskelenmeli.", "{X} dışındaki hiçbir veriye dokunma; sadece bunu etiketle.",
    "Maskeleme kapsamı: {X}. Diğer kişisel veriler görünür kalacak.", "Yalnızca {X} için etiket kullan, başka bir şeyi değiştirme.",
    "{X} gizle, gerisini olduğu gibi bırak.", "Beyaz liste: {X}. Sadece bu alanlar maskelenecek.", "Sadece {X} bilgisini maskele.",
]
BLACK = [
    "{X} haricindeki bütün PII maskelenecek.", "{X} istisna; onun dışındaki tüm kişisel alanları gizle.", "{X} açıkta kalsın; kalan tüm kişisel verileri etiketle.",
    "Her şeyi maskele ama {X} dokunulmadan kalsın.", "Maskeleme yaparken {X} atlanacak, diğer her kişisel veri etiketlenecek.",
    "{X} hariç bütün hassas alanları etiketle.", "{X} dışında kalan tüm kişisel verileri maskele.", "Tüm kişisel verileri maskele fakat {X} olduğu gibi kalsın.",
    "Kara liste: {X}. Bu alan açık kalacak, diğer PII maskelenecek.", "{X} görünür kalsın, geri kalan kişisel verilerin tamamı etiketlensin.",
    "{X} hariç tut; diğer tüm PII alanlarını uygun etiketlerle değiştir.", "Bütün kişisel verileri gizle, yalnızca {X} açık bırak.",
    "{X} maskelenmeyecek; onun dışındaki her kişisel veri etiketlenecek.",
]
GROUP_I = ["Yalnızca {G} maskelenecek.", "Sadece {G} etiketle, diğer alanlar aynen kalsın.", "Bu metinde yalnızca {G} maskele.", "Kapsam: sadece {G}."]

def tag_phrase(tag):
    return f"[{tag}]" if R.random() < 0.4 else R.choice(NAMES[tag])

def join_tr(items):
    if len(items) == 1: return items[0]
    if len(items) == 2: return f"{items[0]} ve {items[1]}"
    return ", ".join(items[:-1]) + " ve " + items[-1]

def pick_kind(long_bias=False):
    r = R.random()
    if r < 0.38: return "duz"
    if r < 0.62: return "kayit"
    if r < 0.86: return "uzun"
    return "cok_kisi"

def make(kategori):
    caps = R.random() < 0.26
    sozle = R.random() < 0.18
    if kategori == "negatif":
        n = R.choice([1, 1, 1, 1, 2, 3])
        segs = []
        for i in range(n):
            if i: segs.append(R.choice([". ", ". ", " "]))
            segs.append(neg_text())
        oz = ["tuzak"]
        r = R.random()
        if r < 0.8: ins = R.choice(FULL)
        elif r < 0.92: ins = R.choice(WHITE).format(X=join_tr([tag_phrase(t) for t in R.sample(TAGS, R.choice([1, 1, 2]))]))
        else: ins = R.choice(BLACK).format(X=join_tr([tag_phrase(t) for t in R.sample(TAGS, R.choice([1, 1, 2]))]))
        mask = set()
    else:
        kind = pick_kind()
        ekli = kind in ("duz", "uzun") and R.random() < 0.22
        segs = build_text(kind, sozle, ekli)
        present = sorted({s[0] for s in segs if isinstance(s, tuple)})
        oz = [kind]
        if ekli: oz.append("ekli")
        if sozle and any(s[0] in SPELLABLE for s in segs if isinstance(s, tuple)): oz.append("sozle")
        if kategori == "tam":
            ins = R.choice(FULL); mask = set(present)
        elif kategori == "grup":
            g = R.choice(list(GROUPS)); ins = R.choice(GROUP_I).format(G=R.choice(GROUP_NAMES[g])); mask = set(present) & set(GROUPS[g])
            kategori = "beyaz_liste"; oz.append("grup")
            if not mask: oz.append("olmayan_etiket_talebi"); kategori = "kapsam_disi"
        elif kategori == "beyaz_liste":
            k = min(len(present), R.choice([1, 1, 1, 2, 2, 3]))
            chosen = R.sample(present, k)
            if R.random() < 0.2:  # mix: one requested tag absent from text
                absent = [t for t in TAGS if t not in present]
                chosen = chosen + [R.choice(absent)]; R.shuffle(chosen); oz.append("olmayan_etiket_talebi")
            ins = R.choice(WHITE).format(X=join_tr([tag_phrase(t) for t in chosen])); mask = set(chosen) & set(present)
        elif kategori == "kara_liste":
            if len(present) < 2 and R.random() < 0.8:
                segs = build_text(R.choice(["kayit", "uzun", "cok_kisi"]), sozle, False); present = sorted({s[0] for s in segs if isinstance(s, tuple)}); oz = [oz[0]]
            k = min(len(present), R.choice([1, 1, 1, 2]))
            excl = R.sample(present, k)
            if R.random() < 0.12:
                absent = [t for t in TAGS if t not in present]; excl = excl + [R.choice(absent)]; R.shuffle(excl); oz.append("olmayan_etiket_talebi")
            ins = R.choice(BLACK).format(X=join_tr([tag_phrase(t) for t in excl])); mask = set(present) - set(excl)
        elif kategori == "kapsam_disi":
            absent = [t for t in TAGS if t not in present]
            chosen = R.sample(absent, R.choice([1, 1, 2]))
            ins = R.choice(WHITE).format(X=join_tr([tag_phrase(t) for t in chosen])); mask = set(); oz.append("olmayan_etiket_talebi")
        else:
            raise ValueError(kategori)
    if caps: oz.append("caps")
    inp, out = render(segs, mask, caps)
    inp, out = re.sub(r"  +", " ", inp).strip(), re.sub(r"  +", " ", out).strip()
    present = sorted({s[0] for s in segs if isinstance(s, tuple)})
    return {"instruction": ins, "input": inp, "output": out, "kategori": kategori, "ozellikler": ",".join(oz),
            "tags": " ".join(f"[{t}]" for t in present), "mask_tags": " ".join(f"[{t}]" for t in sorted(mask))}

MIX = [("tam", 0.32), ("beyaz_liste", 0.19), ("kara_liste", 0.13), ("kapsam_disi", 0.08), ("negatif", 0.24), ("grup", 0.04)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--out", default="data/train.jsonl")
    ap.add_argument("--val_out", default="data/val.jsonl")
    ap.add_argument("--val", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bench", default="data/benchmark_1000.csv")
    ap.add_argument("--max_chars", type=int, default=900)
    args = ap.parse_args()
    R.seed(args.seed)
    bench_inputs = set()
    try:
        import pandas as pd
        bench_inputs = set(pd.read_csv(args.bench).input)
    except Exception as e:
        print("no benchmark for overlap check:", e, file=sys.stderr)
    cats, ws = zip(*MIX)
    seen, rows, stats, overlap = set(), [], Counter(), 0
    while len(rows) < args.n + args.val:
        r = make(R.choices(cats, ws)[0])
        if len(r["input"]) > args.max_chars or r["input"] in seen: continue
        if r["input"] in bench_inputs: overlap += 1; continue
        seen.add(r["input"]); rows.append(r)
        stats[r["kategori"]] += 1
        for o in r["ozellikler"].split(","): stats["oz:" + o] += 1
    R.shuffle(rows)
    with open(args.val_out, "w") as f:
        for r in rows[:args.val]: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.out, "w") as f:
        for r in rows[args.val:]: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)-args.val} train / {args.val} val  (benchmark-overlap dropped: {overlap})")
    for k, v in sorted(stats.items()): print(f"  {k:28s} {v:7d}  {v/len(rows):.3f}")

if __name__ == "__main__":
    main()
