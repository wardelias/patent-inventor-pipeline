"""
Name-based ethnicity classification: 'arab' | 'jewish' | 'ambiguous'.

Implements the three-bucket rule promised to Lia (28/6, point 2): clear Arab
name, clear Jewish name, and a borderline "Israelis" bucket that EXISTS but is
EXCLUDED from the comparison. Only clear-vs-clear enters the regression; the
ambiguous share is reported so we can show the discarded mass is small.

This is a heuristic. It MUST be validated by hand (see validate.py) and the
lexicons below are meant to be extended as you inspect real data.
"""
import re
import unicodedata

# --- Arabic-origin structural markers -------------------------------------
ARAB_PREFIXES = (
    "abu", "abo", "abd", "abed", "abdel", "abdul", "abdal", "ibn", "bin",
    "el", "al", "aal", "ben-el",
)
ARAB_SUFFIXES = ("allah", "ullah", "eddin", "addin", "uddin", "iyeh", "ieh",
                 "awi", "awy", "aoui", "yeh")

# Common Arab-Israeli surnames (Muslim, Christian, Druze, Bedouin).
ARAB_LEXICON = {
    "haddad", "khoury", "khory", "kh oury", "mansour", "mansur", "nassar",
    "nasser", "zoabi", "zuabi", "jabareen", "jabarin", "agbaria", "egbaria",
    "athamneh", "athamna", "masalha", "kanaaneh", "kanaana", "zahalka",
    "shalabi", "hijazi", "awad", "saleh", "salah", "hassan", "hussein",
    "husein", "omar", "mahmoud", "mahmud", "mahameed", "mahamid", "sarsour",
    "bishara", "sabbagh", "sabag", "ghanem", "ghanim", "daoud", "dawood",
    "ibrahim", "yousef", "yusuf", "youssef", "suleiman", "sulaiman",
    "salameh", "salame", "zubeidat", "marzouk", "khatib", "shibli", "amer",
    "sawaed", "tatour", "tatur", "watad", "biadsy", "biadse", "kabha",
    "younis", "yunis", "hamdan", "srour", "srur", "sror", "asadi", "asaad",
    "assad", "haj", "hajj", "hajyahya", "taha", "abbas", "abas", "musa",
    "mousa", "moussa", "issa", "eissa", "isa", "farah", "fares", "faris",
    "hanna", "hana", "jubran", "gibran", "jiryis", "khamis", "khames",
    "kassem", "qasem", "kassim", "naffaa", "nakhleh", "rabah", "rayan",
    "rian", "safadi", "swaid", "suwaid", "tarabeih", "zeidan", "zidan",
    "azzam", "badran", "bakri", "bsoul", "darwish", "diab", "dabbah",
    "fahoum", "falah", "ganem", "grayeb", "gharra", "hamed", "hamid",
    "handal", "haj yahia", "ibraheem", "iraqi", "jaber", "jabr", "kanan",
    "kayal", "kheir", "khalil", "khalaily", "khoraish", "mari", "mari e",
    "matar", "mawasi", "mazawi", "muadi", "muhsen", "mustafa", "naamneh",
    "nahhas", "najjar", "naji", "odeh", "oda", "qadan", "qupty", "raja",
    "rashed", "rashid", "rizik", "rizk", "saad", "sabri", "sader", "safuri",
    "sakas", "salim", "samara", "sayed", "shahin", "shanan", "sharkia",
    "shehadeh", "shihab", "sultani", "tannous", "toameh", "wahbi", "yasin",
    "yassin", "zaher", "zahran", "zarka", "zbeidat", "ziad", "ziadeh",
}

# --- Hebrew/Jewish structural markers -------------------------------------
JEWISH_SUFFIXES = (
    "ovich", "ovitch", "owitz", "ovsky", "ovski", "evsky", "insky", "insku",
    "sky", "ski", "berg", "stein", "shtein", "baum", "feld", "thal", "tal",
    "mann", "man", "sohn", "son", "witz", "wicz", "vitz", "zon", "kind",
    "kraut", "reich", "heim", "berger", "burger", "hauser", "farb", "gold",
)
JEWISH_PREFIXES = ("ben", "bar", "bat", "rosen", "gold", "silber", "gross", "klein")

JEWISH_LEXICON = {
    "cohen", "kohen", "kahn", "kagan", "levy", "levi", "levin", "levine",
    "mizrahi", "mizrachi", "peretz", "biton", "dahan", "avraham", "abraham",
    "friedman", "fridman", "katz", "kaatz", "shapira", "shapiro", "golan",
    "barak", "azoulay", "azulay", "gabay", "gabbay", "malka", "ohana",
    "elbaz", "sharon", "yosef", "yossef", "segal", "sagal", "shani",
    "shalev", "sheffer", "shefer", "adler", "arad", "aviv", "avidan",
    "avni", "bachar", "bahar", "banai", "barzilai", "bassan", "becker",
    "beit", "ben ami", "benayoun", "berkovich", "blum", "brand", "brenner",
    "carmel", "chen", "dagan", "dayan", "dror", "efrat", "eilat", "eitan",
    "elkayam", "erez", "eshel", "even", "gal", "galili", "gavish", "geva",
    "gilad", "ginsburg", "gitlin", "granot", "greenberg", "grinberg",
    "gutman", "harel", "hazan", "hazani", "herman", "hershko",
    "hirsch", "kaplan", "karp", "kessler", "kimchi", "kirsch", "koren",
    "lahav", "lang", "lapid", "lavi", "leibovich", "lerner", "lieberman",
    "livne", "lubin", "maor", "margalit", "meir", "melamed", "mor",
    "nachum", "nahum", "naor", "nave", "nir", "oren", "paz", "pinto",
    "porat", "rabin", "raz", "regev", "reuven", "rom", "ron", "rotem",
    "roth", "rubin", "sagi", "saban", "schwartz", "shaked", "shamir",
    "shapir", "sharabi", "shemesh", "shiloh", "shmueli", "shoham", "sofer",
    "solomon", "shlomo", "spector", "stern", "tal", "tamir", "tzur", "zur",
    "vardi", "weiss", "weisman", "yaakov", "yadin", "yaron", "zaslavsky",
    "zeevi", "zilber", "zohar", "zuckerman",
}

# Observed in the real Israeli inventor data (top-500 unresolved surnames).
JEWISH_LEXICON |= {
    "david", "peled", "peleg", "klein", "ziv", "sela", "alon",
    "lev", "eyal", "bar", "ronen", "yogev", "moshe", "rosen",
    "miller", "dekel", "goren", "shavit", "amit", "gross", "baruch",
    "lifshitz", "aharon", "segev", "frenkel", "raviv", "ashkenazi", "dotan",
    "gur", "israeli", "keren", "zamir", "aharoni", "guy", "arbel",
    "marom", "ram", "shahar", "atias", "daniel", "doron", "diamant",
    "sapir", "navon", "rahamim", "shitrit", "snir", "salomon", "israel",
    "tsur", "carmi", "fuchs", "perry", "raveh", "rapaport", "ilan",
    "gordon", "barel", "lavie", "sadeh", "almog", "drori", "kedem",
    "hillel", "gilboa", "inbar", "magen", "shachar", "caspi", "kogan",
    "landau", "sade", "sinai", "baron", "gil", "haber", "joseph",
    "shemer", "sivan", "leshem", "glazer", "naveh", "ravid", "sella",
    "weiner", "gazit", "aloni", "manor", "or", "shaul", "toledano",
    "shoval", "michaeli", "arnon", "azaria", "sagiv", "kahana", "bitton",
    "livneh", "meiri", "ofir", "green", "oron", "kfir", "romano",
    "haim", "litvak", "peri", "dor", "gonen", "kadosh", "halevi",
    "moyal", "morag", "natan", "ovadia", "shoshani", "yaari", "ashkenazy",
    "peer", "fogel", "geller", "margolin", "schwarz", "bitan", "gabai",
    "kantor", "laufer", "miron", "netzer", "shwartz", "vaknin", "mazor",
    "efrati", "frank", "nathan", "dan", "abramov", "schneider", "yehuda",
    "shor", "ofek", "polak", "einav", "goshen", "oz", "erlich",
    "maoz", "dolev", "harari", "barkan", "ezra", "eliyahu", "fisher",
    "baram", "lotan", "menachem", "cooper", "halfon", "pasternak", "hadar",
    "halperin", "yoffe", "axelrod", "mayer", "zinger", "amitai", "engel",
    "barnea", "eshet", "gafni", "reuveni", "shilo", "shimon", "yehezkel",
    "pollak", "arieli", "kramer", "zigdon", "barkai", "benjamin", "ofer",
    "davidi", "glick", "avrahami", "wexler", "katzir", "yahav", "hollander",
    "mintz", "menashe", "elazar", "armon", "asher", "barash", "brill",
    "kedar", "kushnir", "ohayon", "shaham", "zahavi", "antebi", "harpaz",
    "haviv", "rosenzweig", "saar", "shoshan", "broder", "gurevich", "keshet",
    "margulis", "tadmor", "yaffe", "blank", "singer", "marcus", "zehavi",
    "pinhas", "taub", "ginzburg", "samet", "keidar", "maimon", "rosenblum",
    "rothschild", "yair", "barkay", "yaniv", "benisty", "carmon", "lapidot",
    "meller", "nevo", "adar", "herzog", "hammer", "hacohen", "yanai",
    "mandel", "bloch", "arazi", "tov", "kalderon", "atar", "atzmon",
    "gvili", "yahalom", "ozer", "swissa", "mozes", "banin", "biran",
    "gerber", "horev", "sarig", "shacham", "shur", "zweig", "avner",
    "bareket", "bezalel", "buzaglo", "farber", "finkel", "kehat", "laor",
    "madar", "meron", "nahmias", "oved", "rahav", "rozen", "schreiber",
    "marciano", "markus", "degani", "gershon", "hadas", "heller", "lior",
    "simhon", "eden", "gueta", "hecht", "levanon", "shafir", "tzabari",
    "yarden", "angel", "farhi", "farkash", "gadot", "rozenblat", "shabtai",
    "davidov", "chechik", "sarfati", "dar", "itzhaki", "agam", "giladi",
    "kimhi", "shmuel", "wiener", "fine", "shechter", "lazar", "ayalon",
    "keller", "kotzer", "matza", "weber", "yarkoni", "keinan", "shabat",
    "geffen", "perets", "yuval", "barzilay", "daniely", "assa", "dvir",
    "mashiach", "zobel", "reshef", "reznik", "beeri", "meshulam", "kraus",
    "wolf", "lewin", "rachamim", "savir", "egozi", "blich", "atlas",
    "beck", "kariv", "pomerantz", "guetta", "mordechai", "perez", "almagor",
    "efron", "fishel", "fishler", "honig", "ioffe", "kipnis", "marmor",
    "medina", "pinhasov", "plotkin", "shimoni", "sirkis", "yehezkely", "zak",
    "aflalo", "eisen", "gat", "gefen", "hayun", "kost", "langer",
    "malki", "moran", "nesher", "noy", "saraf", "sharoni", "weingarten",
    "yifrach", "zelig", "rapoport", "aharonov", "averbuch", "hagai", "karni",
    "siegel", "malul", "mesika", "porath", "revach", "sinay", "ophir",
    "abadi", "assouline", "eldar", "eran", "galon", "hadash", "kliger",
    "meidan", "sachs", "tamari", "tavor", "wagner", "barzelay", "shabtay",
    "avneri", "kirshner", "gaon", "zemer", "zucker", "kent", "shiran",
    "ozana", "avigdor", "azriel", "bass", "cohn", "danon", "elad",
    "erel", "fischer", "ganon", "gluck", "hay", "kopel", "mazar",
    "orr", "pfeffer", "soffer", "yellin", "attias", "dvash", "agassi",
    "tirosh", "vax", "fried", "menahem", "richter", "schiller", "schwarcz",
    "shenhar", "brener", "fox", "aviram", "kochavi", "kremer", "shadmi",
    "amram", "ariav", "avivi", "ehrlich", "krauss", "lavee", "maayan",
    "shkedi", "friedlander", "horn", "litvin", "halpern", "badichi", "raab",
    "kasher", "nadav", "sadan", "afek", "sher", "yerushalmi", "ventura",
    "stauber", "winter", "hod", "bahat", "nissan", "noam", "bonen",
    "hertz", "karako", "shlomov", "appel", "glik",
}

# --- Explicitly ambiguous ---------------------------------------------------
# Borne by BOTH Mizrahi/Sephardi Jewish families and Arab families, or
# Western-looking Christian Arab names. These go straight to the third bucket
# regardless of what the other rules say. This is the list Lia's "Israelis"
# category is really for.
AMBIGUOUS_LEXICON = {
    "halabi", "shami", "masri", "masry", "hassoun", "sabbah", "amar",
    "ammar", "elias", "ilias", "george", "georges", "salem", "salm",
    "nasrallah", "shalom", "sultan", "shukri", "haddadi", "yosefi",
    "abu hatzeira", "hakim", "kattan", "katan", "mattar", "michael",
    "musleh", "nader", "naderi", "raad", "sabag", "sahar", "shaul",
    "simon", "toledano", "victor", "zaken", "zakai", "aziz", "adam",
    "asa", "gabriel", "jamil", "kamal", "karim", "nissim", "rafael",
    "samir", "sami", "shakib", "yaacoub", "yacoub", "jacob",
}

AMBIGUOUS_LEXICON |= {
    # Audit findings: Yemenite / Maghrebi / Sephardi Jewish surnames that the
    # Arab rules or lexicon had captured, plus names genuinely used by both.
    "sharaby", "sharabi", "abulafia", "abudi", "meimoun", "maimon",
    "saadoun", "saadon", "atoun", "khayat", "khayyat", "saad", "hakim",
    "shami", "mizrahi", "iraqi", "iraqui", "saudi", "zilkha", "haik",
    "benayoun", "hassoun", "hassoon", "haddad", "hadad",
    "amir", "amar", "hadad", "naim", "hakim", "elias", "sahar", "haham", "assaf", "omer", "michael", "jacob", "simon", "shalom", "nakash", "harush", "shamay", "sorani", "ozeri", "sabag", "salem", "hen", "kadosh",
}
# ambiguous wins over the Jewish list where both fire
JEWISH_LEXICON -= AMBIGUOUS_LEXICON
ARAB_LEXICON -= AMBIGUOUS_LEXICON

# Hand-labelled from arabtown_surnames.csv: surnames carried by inventors
# resident in Arab-majority localities, reviewed by hand rather than by
# threshold. The locality data generated the CANDIDATES; the labels are
# linguistic, because roughly half the single-carrier candidates were Russian
# or Ashkenazi names recorded in an Arab town (Krutyolkin, Schwarzfuchs,
# Statnikov, O'Sullivan) and any automatic cutoff would have swallowed them.
ARAB_LEXICON |= {
    "masarwa", "masarwi", "massarwi", "massarweh", "mahajna", "mhajna", "mahajneh",
    "bshara", "bishara", "copty", "qupty", "ismael", "ismail", "khamaisi",
    "khamaisee", "khamaysi", "khamayseh", "naser", "nasser", "nassir", "attili",
    "badarna", "badarneh", "baransi", "baransy", "basheer", "bashir", "bitar",
    "daas", "daher", "darawshy", "darawshe", "eessa", "elfar", "elkader",
    "farraj", "fawarsy", "ghanaiem", "ghanaim", "ghandour", "grefat", "hassoon",
    "hassoun", "hayik", "hegaze", "hijaze", "housien", "hussien", "jadaon",
    "jbara", "jabara", "khazen", "khshaiboun", "layyous", "mattar", "matar",
    "mazzawi", "miari", "mokatren", "mokthar", "mukhtar", "nimer", "nimre",
    "qaddah", "rahme", "razeq", "riziq", "rizq", "sakran", "salalha",
    "saliba", "sammar", "sarhan", "sirhan", "shajrawi", "shalabney", "sharifie",
    "shihadeh", "shihade", "shomar", "shorosh", "silbaq", "sleem", "takrouri",
    "talhami", "touma", "tuma", "yaseen", "yasseen", "yassin", "zaarura",
    "zeibak", "ziedan", "zoubi", "zuabe", "zubi", "alasibi", "alhowashla",
    "alnabari", "aladin", "benamar", "srouji", "massalha", "masalha", "khawaled",
    "ayoub", "salama", "sharaby", "taher", "haik", "khayat", "saudi",
    "iraqui", "zilkha", "shakour", "shqara", "shqair", "egbariah", "agbariah",
    "jabarin", "jabareen", "abed", "awawdy", "awawdeh", "atamneh", "athamneh",
    "zoabi", "zubeidat", "garra", "gharra", "kabaha", "kabha", "marzouk",
    "nakhle", "nakhleh", "sabbagh", "sarsur", "sarsour", "tarabeih", "watad",
    "zahalka", "zaher", "zeidan", "hamdan", "kanaan", "kanaaneh", "kanana",
    "khalaily", "khoury", "khouri", "mansour", "mansur", "haddad", "hadad",
    "younis", "yunis", "jubran", "gibran", "shibli", "shibly", "asadi",
    "assadi", "swaid", "sawaed", "suwaid", "dabbah", "dabbagh", "diab",
    "darwish", "falah", "fahoum", "grayeb", "handal", "jaber", "jabr",
    "kayal", "kheir", "khalil", "naamneh", "nahhas", "najjar", "odeh",
    "oda", "qadan", "rashed", "rashid", "sabri", "sader", "safuri",
    "shahin", "shanan", "shehadeh", "shihab", "tannous", "toameh", "wahbi",
    "zahran", "zarka", "ziadeh", "biadsy", "biadse", "boulos", "butros",
    "ghattas", "hanna", "hana", "jiryis", "kassab", "khamis", "mansor",
    "nassrallah", "sahouri", "sayegh", "shammas", "shoufani", "tabari", "zoubir",
    "moadi", "muadi", "kablan", "kaadan", "kadan", "hamoud", "hammoud",
    "osman", "othman", "rayan", "rian", "sabag", "safadi",
}
ARAB_LEXICON -= AMBIGUOUS_LEXICON
JEWISH_LEXICON -= ARAB_LEXICON

_STRIP = re.compile(r"[^a-z\s-]")


def normalize(name: str) -> str:
    """Lowercase, de-accent, collapse separators."""
    if not isinstance(name, str):
        return ""
    n = unicodedata.normalize("NFKD", name)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.lower().strip()
    n = _STRIP.sub(" ", n)
    n = re.sub(r"[\s-]+", " ", n).strip()
    return n


def _tokens(n: str):
    return [t for t in n.split(" ") if t]


def classify(surname: str) -> str:
    """Return 'arab', 'jewish', or 'ambiguous'."""
    n = normalize(surname)
    if not n or len(n) < 2:
        return "ambiguous"

    joined = n.replace(" ", "")
    toks = _tokens(n)

    # DECISIVE: an Arabic patronymic construction is conclusive no matter what
    # the following element is. Checked FIRST, before the borderline lexicon --
    # otherwise "Abu Salem" and "El Hakim" were binned as borderline because
    # 'salem' and 'hakim' are borderline as standalone surnames.
    #
    # Guarded by a full-name lexicon check: some Jewish surnames begin with
    # these particles (Abu Hatzeira, a Moroccan rabbinical family; Abulafia;
    # Abudi). An explicit entry for the WHOLE name always wins.
    _whole = (n in AMBIGUOUS_LEXICON or joined in AMBIGUOUS_LEXICON
              or n in JEWISH_LEXICON or joined in JEWISH_LEXICON)
    if not _whole and len(toks) > 1 and toks[0] in ("abu", "abo", "abd", "abed", "abdel",
                                                                    "abdul", "abdal", "ibn", "bin",
                                                                    "al", "el", "aal"):
        return "arab"
    if not _whole and joined.startswith(("abdel", "abdul", "abdal", "abdur",
                                         "abdo")):
        return "arab"

    # Hard override: known both-communities names.
    if n in AMBIGUOUS_LEXICON or joined in AMBIGUOUS_LEXICON:
        return "ambiguous"
    if any(t in AMBIGUOUS_LEXICON for t in toks):
        return "ambiguous"

    # Lexicon hits are DEFINITIVE and beat structural guesses. Without this,
    # the greedy Jewish "-man" suffix stole Osman, Othman and Suleiman.
    lex_arab = (n in ARAB_LEXICON or joined in ARAB_LEXICON
                or any(t in ARAB_LEXICON for t in toks))
    lex_jew = (n in JEWISH_LEXICON or joined in JEWISH_LEXICON
               or any(t in JEWISH_LEXICON for t in toks))
    if lex_arab and not lex_jew:
        return "arab"
    if lex_jew and not lex_arab:
        return "jewish"
    if lex_arab and lex_jew:
        return "ambiguous"

    # No lexicon hit -- fall back to structural markers.
    arab = jewish = False
    if toks and toks[0] in ARAB_PREFIXES and len(toks) > 1:
        arab = True
    # "abu"/"abd" must be a SEPARATE token. As a bare prefix it fired on
    # Abulafia and Abudi, both Sephardi Jewish surnames.
    # An Arabic patronymic prefix is DECISIVE and returns immediately. It
    # outranks the weak Jewish "-man" suffix, which was cancelling it out
    # (abdelrahman ends in -man, so Abdel Rahman came back ambiguous).
    if toks and toks[0] in ("abu", "abo", "abd", "abed", "abdel", "abdul",
                            "abdal", "ibn", "bin"):
        return "arab"
    if joined.startswith(("abdel", "abdul", "abdal", "abdur", "abdo")):
        return "arab"
    if joined.endswith(ARAB_SUFFIXES):
        arab = True

    if toks and toks[0] in JEWISH_PREFIXES and len(toks) > 1:
        jewish = True
    if joined.endswith(JEWISH_SUFFIXES):
        jewish = True

    if arab and jewish:
        return "ambiguous"
    if arab:
        return "arab"
    if jewish:
        return "jewish"
    return "ambiguous"


def classify_series(s):
    """Vectorised over a pandas Series of surnames."""
    return s.map(classify)


# ---------------------------------------------------------------------------
# Learning surnames from residential segregation
# ---------------------------------------------------------------------------
# The hand lexicons cover common names. The tail is thousands of rare surnames
# that no list will ever reach. So we learn them from the data.
#
# Israeli localities are highly segregated: Sakhnin, Tamra, Umm al-Fahm, Rahat
# and similar are ~100% Arab; Kfar Saba, Herzliya and similar are ~100% Jewish.
# A surname carried mostly by inventors living in Arab localities is an Arab
# surname. We learn the LEXICON from locality, then apply it by NAME -- so an
# Arab inventor living in Tel Aviv is still classified correctly.
#
# CAVEATS to state in the paper:
#  - Mixed cities (Haifa, Acre, Lod, Ramla, Jerusalem) are excluded from the
#    learning step, so they contribute no evidence either way.
#  - Learning uses city while the regression controls for region. The learned
#    object is a name->ethnicity map, not a city indicator, and the district
#    control is coarser than locality -- but report a robustness run with the
#    hand lexicons only.
#  - Requires MIN_N carriers before a surname is assigned, so rare names stay
#    ambiguous rather than being guessed from one person.

LEARN_MIN_N = 99999   # learner DISABLED -- see LEARNER NOTE
LEARN_HI = 0.99       # >= this share -> Arab. 0.99 = effectively OFF;
                      # see LEARNER NOTE below.
LEARN_LO = 0.00       # <= this share -> Jewish surname (OFF by default)
LEARN_ASSIGN_JEWISH = False

# LEARNER NOTE  (updated after the city audit)
# The learner is now fully disabled. The audit showed the city field is not
# merely noisy but systematically wrong: localities such as Rahat, Kalanswa
# and Sachnin carry far more inventors than their size supports, and only
# 3-12% of those inventors have Arab surnames, against 60-75% in Nazareth,
# Taibe and Umm al-Fahm. This is consistent with geocoding that snaps
# inventors to a nearby locality centroid. Learning surnames from such data
# imports the error directly into the lexicon -- it produced Porgador and
# Bachmat, both Jewish, as "Arab" surnames.
#
# ORIGINAL NOTE
# The locality learner is effectively disabled (LEARN_HI = 0.99). At min_n=1 it
# assigned "Arab" to roughly half of the single-carrier candidates, which were
# Russian and Ashkenazi surnames recorded in Arab towns. At min_n>=3 it found
# almost nothing, because only ~700 inventors live in Arab localities at all
# and 84% of their surnames are unique. Neither setting is usable.
# Localities were therefore used to GENERATE candidate surnames, which were
# then labelled by hand into ARAB_LEXICON below. That is the defensible
# version of the method and the one to describe in the paper.

# WHY THE JEWISH SIDE IS OFF
# Only ~2% of Israeli inventors live in Arab-majority localities. A rule like
# "<=5% of carriers live in Arab towns -> Jewish" therefore sits barely above
# the base rate and fires for almost every surname, including Arab surnames
# whose carriers happen to live in Haifa or Jerusalem. In the first run this
# pushed 294 residents of Arab towns into the Jewish bucket and contaminated
# the control group. The hand lexicon is strong on the Jewish side already;
# the learner exists to reach ARAB names, so that is all it now does.


def learn_from_localities(surnames, cities, min_n=LEARN_MIN_N,
                          hi=LEARN_HI, lo=LEARN_LO,
                          assign_jewish=LEARN_ASSIGN_JEWISH, verbose=True):
    """
    surnames, cities: equal-length sequences, one entry per INVENTOR.
    Returns {normalised_surname: 'arab'|'jewish'}.
    """
    import collections
    import regions as rg

    tot = collections.Counter()
    arab = collections.Counter()
    for s, c in zip(surnames, cities):
        n = normalize(s)
        if not n:
            continue
        tot[n] += 1
        if rg.is_arab_locality(c):
            arab[n] += 1

    learned = {}
    for n, t in tot.items():
        if t < min_n:
            continue
        frac = arab[n] / t
        if frac >= hi:
            # never override an unambiguous structural Jewish signal
            if classify(n) != "jewish":
                learned[n] = "arab"
        elif assign_jewish and frac <= lo:
            learned[n] = "jewish"

    if verbose:
        na = sum(v == "arab" for v in learned.values())
        nj = sum(v == "jewish" for v in learned.values())
        print(f"    learned from localities: {na:,} Arab surnames, "
              f"{nj:,} Jewish surnames "
              f"(from {len(tot):,} distinct names, min_n={min_n})")
    return learned


def classify_series(s, learned=None):
    """
    Vectorised classification. If `learned` is supplied it takes precedence,
    because it is name-specific evidence from the data rather than a guess
    from a hand list.
    """
    if not learned:
        return s.map(classify)

    def one(x):
        n = normalize(x)
        if n in learned:
            return learned[n]
        return classify(x)

    return s.map(one)
