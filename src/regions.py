"""
Inventor city -> region, plus a flag for Arab-majority localities.

Two jobs now:
  1. to_district(city)      -> CBS district, for the Region control
  2. is_arab_locality(city) -> used by names.py to LEARN which surnames are
                               Arab, from the data itself

Israeli localities are highly residentially segregated, which is what makes
job 2 work. See names.learn_from_localities for the method and its caveats.

v2: fuzzy matching for spelling variants (Kfar Sava -> Kfar Saba, Tveria ->
Tiberias, Gdera -> Gedera), many more localities, Arab-locality set.
"""
import difflib
import re
import unicodedata

_STRIP = re.compile(r"[^a-z0-9\s]")
_NOISE = ("israel", "city", "moshav", "kibbutz")


def normalize(s: str) -> str:
    if not isinstance(s, str):
        return ""
    n = unicodedata.normalize("NFKD", s)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.lower().strip().replace("-", " ").replace("'", "").replace("`", "")
    n = _STRIP.sub(" ", n)
    return re.sub(r"\s+", " ", n).strip()


def _canon(s: str) -> str:
    """Aggressive form for fuzzy matching: drop noise words, collapse vowels."""
    n = normalize(s)
    for w in _NOISE:
        n = n.replace(w, "")
    n = re.sub(r"\s+", " ", n).strip()
    n = n.replace("kfar", "kefar").replace("qiryat", "kiryat")
    n = n.replace("bet ", "beit ").replace("ts", "tz").replace("kh", "h")
    n = re.sub(r"[aeiou]+", "a", n)
    return n


ARAB_LOCALITIES = {
    "nazareth", "nazerat", "natzrat", "yafa an naseriyye", "yafia", "reineh",
    "iksal", "kafr kanna", "kfar kanna", "mashhad", "ein mahel", "ein mahil",
    "kafr manda", "bueine nujeidat", "bueine", "shibli", "umm al ghanam",
    "daburiya", "tamra", "kabul", "ibillin", "shefaram", "shfaram",
    "shefa amr", "sakhnin", "sachnin", "arraba", "arrabe", "deir hanna",
    "eilabun", "majd al krum", "majdal krum", "nahf", "rameh", "rama",
    "sajur", "jish", "judeida", "maker", "judeida maker", "kafr yasif",
    "abu snan", "julis", "yarka", "beit jann", "hurfeish", "kisra", "sumei",
    "kisra sumei", "maghar", "mghar", "peki in", "bir al maksur",
    "basmat tabun", "zarzir", "kaabiyye", "tuba zangariyye", "ghajar",
    "umm al fahm", "umm el fahm", "baqa al gharbiyye", "baqa al gharbiyya",
    "jatt", "zemer", "kafr qara", "kfar kara", "ar ara", "arara",
    "muawiya", "basma", "jisr az zarqa", "fureidis", "furadis",
    "daliyat al karmel", "daliat el carmel", "isfiya", "usfiya", "tira",
    "tayibe", "taibe", "tayiba", "qalansawe", "kalanswa", "kalansuwa",
    "kafr qasim", "kafr kassem", "kafr bara", "jaljulia", "jaljulya",
    "rahat", "hura", "lakiya", "kuseife", "kseife", "tel sheva",
    "tel as sabi", "segev shalom", "shaqib al salam", "ar arah banegev",
    "arara banegev", "abu ghosh", "ein rafa", "ein naqquba", "al araqeeb",
    "araqib", "sandala", "muqeible", "zalafa", "musheirifa",
    "deir al asad", "deir el asad", "biane", "bina", "bineh", "tal as sabi",
    "hora", "yafia", "yafa an nasiriyya",
}

DISTRICTS = {
    "north": [
        "nazareth", "nazerat", "natzrat", "nof hagalil", "nazareth illit",
        "afula", "tiberias", "teverya", "tveria", "safed", "tzfat", "zefat",
        "karmiel", "carmiel", "acre", "akko", "nahariya", "nahariyya",
        "nahariah", "maalot", "tarshiha", "maalot tarshiha", "shefaram",
        "shfaram", "shefa amr", "sakhnin", "sachnin", "arraba", "arrabe",
        "deir hanna", "kafr kanna", "kfar kanna", "kafr manda", "tamra",
        "iksal", "yafia", "reineh", "rameh", "majd al krum", "majdal krum",
        "bueine", "kabul", "judeida", "maker", "jish", "maghar", "mghar",
        "beit jann", "yarka", "julis", "kisra", "hurfeish", "peki in",
        "rama", "eilabun", "kfar yasif", "abu snan", "migdal haemek",
        "migdal ha emek", "yokneam", "yoqneam", "yokneam illit",
        "yoqneam illit", "kiryat shmona", "kiryat tivon", "katzrin",
        "rosh pina", "hatzor", "beit shean", "bet shean", "kfar tavor",
        "kfar vradim", "shlomi", "nahf", "ein mahel", "mashhad", "shibli",
        "zarzir", "bir al maksur", "basmat tabun", "tuba zangariyye",
        "kaabiyye", "sajur", "sumei", "timrat", "nofit", "shimshit",
        "givat ela", "hoshaya", "alon hagalil", "gilon", "manof",
        "yuvalim", "eshhar", "hararit", "moreshet", "mitzpe aviv", "segev",
        "yodfat", "ein hod", "kfar hasidim", "ramat yishai", "ibillin",
        "deir al asad", "deir el asad", "biane", "bina", "yafia",
        "rakefet", "koranit", "amirim", "ram on", "kerem maharal", "amikam",
        "katzir", "harduf", "shorashim", "tuval", "lotem", "kishorit",
        "daburiya", "ghajar",
    ],
    "haifa": [
        "haifa", "hefa", "hadera", "khadera", "kiryat ata", "kiryat atta",
        "kiryat bialik", "kiryat motzkin", "kiryat yam", "nesher",
        "tirat carmel", "tirat hacarmel", "daliyat al karmel",
        "daliat el carmel", "isfiya", "usfiya", "furadis", "fureidis",
        "jisr az zarqa", "zichron yaakov", "zikhron yaakov",
        "zichron ya akov", "binyamina", "pardes hanna", "karkur",
        "or akiva", "caesarea", "kesariya", "atlit", "rekhasim", "harish",
        "givat ada", "hogla", "umm al fahm", "umm el fahm",
        "baqa al gharbiyye", "jatt", "kafr qara", "kfar kara", "ar ara",
        "arara", "basma", "zemer", "ein carmel", "maagan michael",
        "sdot yam", "beit oren", "muawiya", "zalafa", "musheirifa",
    ],
    "tel aviv": [
        "tel aviv", "tel aviv yafo", "tel aviv jaffa", "jaffa", "yafo",
        "ramat gan", "givatayim", "bnei brak", "bene beraq", "holon",
        "bat yam", "herzliya", "herzlia", "ramat hasharon", "or yehuda",
        "kiryat ono", "yehud", "yehud monosson", "azor", "givat shmuel",
    ],
    "center": [
        "petah tikva", "petach tikva", "rishon lezion", "rishon le zion",
        "netanya", "nethanya", "rehovot", "rechovot", "ness ziona",
        "nes ziona", "kfar saba", "kfar sava", "kefar sava", "raanana",
        "hod hasharon", "rosh haayin", "rosh ha ayin", "lod", "ramla",
        "ramle", "modiin", "modi in", "shoham", "elad", "yavne", "gedera",
        "gdera", "kiryat ekron", "mazkeret batya", "gan yavne", "tira",
        "tayibe", "taibe", "kafr qasim", "kafr bara", "jaljulia",
        "qalansawe", "kalanswa", "tel mond", "even yehuda", "kadima",
        "zoran", "pardesiya", "beit dagan", "savyon", "ganei tikva",
        "kfar yona", "beer yaakov", "bat hefer", "matan", "ein vered",
        "zur moshe", "shetulim", "lapid", "neve yaraq", "nof ayalon",
        "kfar vitkin", "kfar yehoshua", "kochav yair", "tzur yigal",
        "kochav yair tzur yigal", "nirit", "sde warburg", "udim",
        "rishpon", "bnei zion", "kfar netter", "harutzim", "tzofit",
        "kfar malal", "magshimim", "nordiya", "hagor",
        # from the audit's unmapped list
        "ramot hashavim", "michmoret", "kfar bilu", "kfar shmaryahu",
        "kfar hess", "kfar sirkin", "kfar truman", "kidron", "kfar refael",
        "karmei yosef", "yad binyamin", "beit hashmonai", "beit hashmonay",
        "nir tzvi", "ganot", "bareket", "zeitan", "yagel", "mishmar hashiva",
    ],
    "jerusalem": [
        "jerusalem", "yerushalayim", "beit shemesh", "bet shemesh",
        "mevaseret zion", "mevaseret tsiyon", "maoz ziyyon", "maale adumim",
        "givat zeev", "abu ghosh", "tzur hadassah", "kiryat yearim",
        "ein rafa", "ein naqquba", "har adar", "efrat",
    ],
    "south": [
        "beer sheva", "beersheba", "be er sheva", "ashdod", "ashkelon",
        "ashqelon", "eilat", "elat", "dimona", "arad", "netivot", "ofakim",
        "sderot", "kiryat gat", "kiryat malachi", "yeruham", "yeroham",
        "mitzpe ramon", "mizpe ramon", "rahat", "tel sheva", "hura",
        "lakiya", "segev shalom", "kuseife", "kseife", "arara banegev",
        "shaqib al salam", "omer", "meitar", "lehavim", "al araqeeb",
        "tal as sabi", "tal sheva", "hora",
        "sde boker", "revivim", "urim", "nitzan",
    ],
}

COARSE = {"north": "north", "haifa": "north", "tel aviv": "center",
          "center": "center", "jerusalem": "center", "south": "south"}

_LOOKUP, _CANON = {}, {}
for _d, _cities in DISTRICTS.items():
    for _c in _cities:
        _LOOKUP[normalize(_c)] = _d
        _CANON.setdefault(_canon(_c), _d)
_CANON_KEYS = list(_CANON)

_ARAB_NORM = {normalize(c) for c in ARAB_LOCALITIES}
_ARAB_CANON = {_canon(c) for c in ARAB_LOCALITIES}
_ARAB_KEYS = list(_ARAB_CANON)

# LOCALITIES WITH UNRELIABLE GEOCODING
# The audit found these carry far more inventors than their population
# supports, and only 3-15% of those inventors have Arab surnames -- against
# 60-75% in Nazareth, Taibe, Umm al-Fahm and Kfar Kanna. The pattern is
# consistent with inventors in nearby Jewish towns being snapped to these
# centroids. Any locality-based measure of ethnicity should exclude them.
UNRELIABLE_GEOCODE = {
    "rahat", "kalanswa", "qalansawe", "sachnin", "sakhnin", "mashhad",
    "ein mahel", "ein mahil", "jaljulya", "jaljulia", "kfar kara",
    "kafr kara", "kfar qara", "al araqeeb", "araqib", "anata",
}
_UNRELIABLE_N = {normalize(x) for x in UNRELIABLE_GEOCODE}


def geocode_reliable(city: str) -> bool:
    """False where the audit found systematic geocoding error."""
    return normalize(city) not in _UNRELIABLE_N


_dcache, _acache = {}, {}


def to_district(city: str) -> str:
    """Six CBS districts, or 'unknown'. Exact, then canonical, then fuzzy."""
    n = normalize(city)
    if not n:
        return "unknown"
    if n in _dcache:
        return _dcache[n]

    d = _LOOKUP.get(n) or _CANON.get(_canon(n))
    if d is None:
        m = difflib.get_close_matches(_canon(n), _CANON_KEYS, n=1, cutoff=0.86)
        if m:
            d = _CANON[m[0]]
    if d is None:
        hits = [(len(k), v) for k, v in _LOOKUP.items()
                if len(k) > 4 and (k in n or n in k)]
        d = max(hits)[1] if hits else "unknown"

    _dcache[n] = d
    return d


# Jewish localities whose names fuzzy-match Arab ones after vowel collapse.
# Timrat/Tamra was a real false positive. A false positive here is far worse
# than a miss: it injects Jewish surnames into the learned Arab lexicon.
NOT_ARAB = {
    "anata", "anatot", "almon",   # West Bank / not Israeli-Arab localities
    # Nazareth Illit (Nof HaGalil) is a majority-JEWISH city adjacent to
    # Nazareth. _canon() strips "illit" as noise and collapsed the two.
    "nazareth illit", "nazareth ilit", "nazerat illit", "natzrat illit",
    "nof hagalil", "nof ha galil",
    "timrat", "tamar", "tirat carmel", "tirat hacarmel", "tirat zvi",
    "tirat yehuda", "ramat yishai", "nofit", "shimshit", "gilon", "manof",
    "segev", "yodfat", "hoshaya", "ramat gan", "kfar hasidim", "sde boker",
    "arad", "omer", "meitar", "lehavim", "yeruham", "yeroham", "harish",
    "givat ada", "kfar tavor", "kfar vradim", "moreshet", "mitzpe aviv",
}
# NB: Kfar Kara, Hura and Arara are genuine Arab towns. Never blocklist them.
_NOT_ARAB_N = {normalize(x) for x in NOT_ARAB}
_NOT_ARAB_C = {_canon(x) for x in NOT_ARAB}


def is_arab_locality(city: str) -> bool:
    """
    True if the city is an Arab-majority Israeli locality.

    Deliberately conservative: exact and canonical matches, then fuzzy only at
    a high cutoff, and never for a name on the NOT_ARAB blocklist. Missing a
    genuine Arab town costs a little recall in the learning step; wrongly
    flagging a Jewish town corrupts the learned lexicon.
    """
    n = normalize(city)
    if not n:
        return False
    if n in _acache:
        return _acache[n]

    if n in _NOT_ARAB_N or _canon(n) in _NOT_ARAB_C:
        r = False
    elif n in _ARAB_NORM or _canon(n) in _ARAB_CANON:
        r = True
    else:
        r = bool(difflib.get_close_matches(_canon(n), _ARAB_KEYS, n=1, cutoff=0.94))

    _acache[n] = r
    return r


def to_coarse(district: str) -> str:
    return COARSE.get(district, "unknown")


def coverage_report(cities):
    from collections import Counter
    return Counter(c for c in cities if to_district(c) == "unknown").most_common()
