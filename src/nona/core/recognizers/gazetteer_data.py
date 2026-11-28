"""Seed lists for the gazetteer recognizer (deep dive A, layer 3). Data only, no logic.

Written independently of the benchmark generator's seed lists (`nona.bench.generate`), from common names in
Tunisia and the Maghreb, plus frequent French and English first names. Overlap with the generator is expected
for the most common names, so the PR that adds a list reports the benchmark both with and without the
generator's names (an ablation), to keep the dev number honest.

Entries are written as people write them. Spelling variants that `core/canonical.py` already unifies
(`Mohammed` → `mohamed`, `3ali` → `ali`, Arabic `سامي` → `sami`) need only one entry.

`COMMON_WORDS` are names that are also ordinary words (`Amal` = hope, `Nour` = light, `Anis` = anise,
`Gharbi` = western). They fire only with a capital letter (0.6 mid-sentence, 0.5 at the start of a sentence,
which the default threshold still masks), or after a cue.
"""

from __future__ import annotations

FIRST_NAMES_TN: tuple[str, ...] = (
    # men
    "Abdallah", "Abdelaziz", "Abdelkader", "Abdelkarim", "Abdelmajid", "Abdelwahed", "Adel", "Adem", "Ahmed",
    "Ala", "Ali", "Amine", "Ammar", "Amor", "Anis", "Anouar", "Ayoub", "Aymen", "Aziz", "Bassem", "Bechir",
    "Belgacem", "Bilel", "Borhen", "Brahim", "Chaker", "Chedly", "Chokri", "Dhia", "Elyes", "Fadhel", "Fares",
    "Farouk", "Fathi", "Faycel", "Fedi", "Firas", "Ghassen", "Habib", "Haythem", "Hamza", "Hassen", "Hatem",
    "Hedi", "Helmi", "Hichem", "Houssem", "Ilyes", "Imed", "Iheb", "Issam", "Jalel", "Jamel", "Jaouhar",
    "Karim", "Kamel", "Khaled", "Khalil", "Lassaad", "Lotfi", "Mahdi", "Mahmoud", "Majdi", "Makram", "Malek",
    "Marouane", "Maher", "Mehdi", "Mohamed", "Mokhtar", "Moncef", "Montassar", "Mondher", "Mounir", "Mourad",
    "Mustapha", "Nabil", "Naceur", "Nader", "Nizar", "Noureddine", "Omar", "Oussama", "Raouf", "Ridha",
    "Riadh", "Rached", "Radhouane", "Rafik", "Ramzi", "Saber", "Sabri", "Sadok", "Safouane",
    "Sahbi", "Said", "Salah", "Salem", "Sami", "Seif", "Selim", "Skander", "Slim", "Slah", "Sofiene",
    "Sofien", "Tarek", "Taoufik", "Wajdi", "Walid", "Wassim", "Yacine", "Yassine", "Yahya", "Youssef", "Zied",
    "Zouhair",
    # women
    "Aicha", "Alya", "Amal", "Amira", "Amel", "Arij", "Asma", "Azza", "Baya", "Besma", "Cyrine", "Dorra",
    "Emna", "Eya", "Faten", "Fatma", "Ghada", "Hajer", "Hanen", "Hela", "Hiba", "Houda", "Imen", "Ines",
    "Jihen", "Khadija", "Kaouther", "Latifa", "Leila", "Lina", "Lobna", "Manel", "Mariem", "Maroua",
    "Meriem", "Mouna", "Monia", "Nadia", "Najet", "Nawel", "Nesrine", "Nour", "Nourhene", "Olfa", "Ons",
    "Rahma", "Rania", "Rim", "Rihab", "Rym", "Sabrine", "Safa", "Salma", "Samia", "Sana", "Sarra", "Siwar",
    "Sonia", "Souad", "Syrine", "Takwa", "Wafa", "Wiem", "Yosra", "Zeineb", "Zohra",
    # added with #50's common-word list
    # (Yasmine is on the owner's list too, but it is a held-out bench name, `tn2`: it stays unlisted)
    "Hayet", "Farah", "Jamila", "Naima", "Kmar", "Chams", "Warda", "Nessma", "Aziza", "Habiba",
)  # fmt: skip

FIRST_NAMES_FR_EN: tuple[str, ...] = (
    "Alexandre", "Anne", "Antoine", "Camille", "Charlotte", "Chloé", "Claire", "Élodie", "Émilie", "Florian",
    "François", "Guillaume", "Hugo", "Isabelle", "Julie", "Julien", "Laura", "Léa", "Louis", "Lucas", "Manon",
    "Margaux", "Marie", "Mathieu", "Maxime", "Nathalie", "Nicolas", "Pauline", "Philippe", "Pierre", "Romain",
    "Sophie", "Valérie", "Vincent",
    "Andrew", "Benjamin", "Charlie", "Daniel", "David", "Emily", "Emma", "Ethan", "George", "Grace", "Hannah",
    "Jack", "James", "Jessica", "John", "Lucy", "Megan", "Michael", "Oliver", "Olivia", "Ryan",
    "Samuel", "Sarah", "Sophia", "Thomas", "William",
)  # fmt: skip

SURNAMES_TN: tuple[str, ...] = (
    "Abidi", "Ammar", "Arfaoui", "Ayadi", "Ayari", "Azzouz", "Baccar", "Bahri", "Bakir", "Belaid", "Belhadj",
    "Ben Abdallah", "Ben Ahmed", "Ben Ali", "Ben Amor", "Ben Aissa", "Ben Brahim", "Ben Hassen",
    "Ben Mansour", "Ben Miled", "Ben Mustapha", "Ben Nasr", "Ben Othman", "Ben Romdhane", "Ben Said",
    "Ben Salah", "Ben Salem", "Ben Slimane", "Ben Youssef", "Bouaziz", "Bouazizi", "Boughanmi", "Bouslama",
    "Chaabane", "Chaouch", "Chebbi", "Cherif", "Dhaouadi", "Dridi", "Ellouze", "Fakhfakh", "Ferchichi",
    "Gabsi", "Gharbi", "Ghannouchi", "Guesmi", "Hadj Ali", "Hamdi", "Hammami", "Hamrouni", "Hosni",
    "Jaziri", "Jebali", "Jelassi", "Jlassi", "Jendoubi", "Jouini", "Kallel", "Karoui", "Kchaou", "Khelifi",
    "Khemiri", "Ksouri", "Ladhari", "Laabidi", "Labidi", "Mabrouk", "Maaloul", "Mansouri", "Marzouki",
    "Masmoudi", "Mathlouthi", "Mbarek", "Mejri", "Messaoudi", "Mezghani", "Miladi", "Nasri", "Nefzi",
    "Ouali", "Ouertani", "Rekik", "Riahi", "Romdhani", "Saadi", "Saidi", "Sassi", "Selmi", "Sfaxi", "Slimi",
    "Snoussi", "Souissi", "Tlili", "Toumi", "Trabelsi", "Triki", "Yahyaoui", "Zaier", "Zouari", "Zribi",
)  # fmt: skip

# Arabic script. Each has its Latin form in `canonical.VARIANTS`, so it links across scripts (tested).
NAMES_AR: tuple[str, ...] = (
    "أحمد", "محمد", "علي", "سامي", "يوسف", "محمود", "عمر", "حمزة", "كريم", "مهدي", "أمين", "ياسين", "خالد",
    "هشام", "وليد", "منصف", "نبيل", "طارق", "عبدالله", "صالح", "سليم", "فاطمة", "خديجة", "مريم", "أمال",
    "نور", "ليلى", "سلمى", "رحمة", "عائشة", "زينب", "إيناس", "ريم", "أسماء", "هدى", "سارة", "منى", "نادية",
    "الطرابلسي", "الغربي", "الجلاصي", "المنصوري", "بن سالم", "بن علي", "بن عمر", "الخليفي", "الماجري",
)  # fmt: skip

COMMON_WORDS: tuple[str, ...] = (
    "Amal", "Nour", "Salam", "Sabri", "Anis", "Salem", "Said", "Malek", "Ali", "Baya", "Ons", "Wafa", "Safa",
    "Hiba", "Rahma", "Lina", "Eya", "Alya", "Grace", "Jack", "Gharbi", "Sfaxi", "Bahri", "Cherif", "Mabrouk",
    "Ammar", "Amor", "Laura", "Charlie", "Ala", "أمال", "نور", "رحمة", "سالم", "صالح",
    # Darija and Arabic words that are also names: `sana` (year), `salah` (prayer), `aziz` (dear),
    # `karim` (generous), `amel`/`3amel` (doing), `fares` (rider), `kamel` (whole), `hassen` (better),
    # `adel` (fair), `amine` (trustworthy), `habib` (dear), `hedi` (calm), `imen` (faith), `houda`
    # (guidance), `rim` (gazelle), `faten` (charming), `selim`/`slim` (safe), `ridha` (consent),
    # `najet` (rescue), `jamel` (beauty), `nabil` (noble), `khaled` (eternal), `sabrine` (patient)
    "Sana", "Salah", "Aziz", "Karim", "Amel", "Fares", "Kamel", "Hassen", "Adel", "Amine", "Habib", "Hedi",
    "Imen", "Houda", "Rim", "Faten", "Selim", "Slim", "Ridha", "Najet", "Jamel", "Nabil", "Khaled",
    "Sabrine", "Mounir", "Monia", "Hanen", "Latifa", "Jalel",
    # `sahbi` (my friend), `saber` (patient), `seif` (summer, sword), `anouar` (lights)
    "Sahbi", "Saber", "Seif", "Anouar",
    # #50, validated by the owner: `hayet` (life), `farah` (joy), `jamila` (beautiful), `naima` (bliss),
    # `kmar` (moon), `chams` (sun), `warda` (rose), `nessma` (breeze), `dorra` (pearl), `amira` (princess),
    # `bechir` (bearer of good news), `aziza` (dear), `habiba` (beloved). `yasmine` (jasmine) is a held-out
    # bench name, so it isn't listed at all.
    "Hayet", "Farah", "Jamila", "Naima", "Kmar", "Chams", "Warda", "Nessma", "Dorra", "Amira", "Bechir",
    "Aziza", "Habiba",
)  # fmt: skip

# Context lists of #50, validated by the owner (a native Darija speaker) on 2026-11-22. Arabizi digits:
# 3 = ع, 7 = ح, 9 = ق, 5 = خ. Matched on the normalized view; Arabic spelling variants (hamza on alef, ta
# marbuta or heh, alef maqsura or yeh) are handled by the recognizer.

# Kinship and relations: a person comes next. Strong cues in Latin letters, where a capital still has to
# start an unlisted name.
KIN_CUES: tuple[str, ...] = (
    "khouya", "khou", "okhti", "o5ti", "ou5ti", "5ti", "sa7bi", "sa7bti", "s7abi", "jari", "jarti", "mart",
)  # fmt: skip
# These lift listed names only, like the speech verbs. The Arabic forms: Arabic has no capitals to tell a
# name from a word, and `جاري` is also "current" (`حساب جاري`), `ولد` also "born" (review of #97). `rajel`
# and `weld` also precede adjectives (`rajel Tounsi`); they are kept knowing that `rajel karim` (a generous
# man) and `weld 3aziz` (a dear boy) get masked.
KIN_CUES_LISTED: tuple[str, ...] = (
    "rajel", "weld",
    "خويا", "أختي", "ختي", "صاحبي", "صاحبتي", "جاري", "جارتي", "مرت", "راجل", "ولد",
)  # fmt: skip

# Speech and contact verbs, with their preposition: they lift a listed name only (owner's choice, 2026-11-28),
# so `kallamt Ooredoo` stays unmasked. `chouft` is kept knowing `chouft amal` (I saw hope) gets masked. In
# Arabic the `ل` may be glued to the name (`قلت لكريم`): the recognizer strips it as a clitic.
SPEECH_CUES: tuple[str, ...] = (
    "9olt l", "9otlou", "9al l", "kallamt", "kallemt", "kalamt", "3ayatt l", "3ayyatt l", "telfent l",
    "b3atht l", "ba3tht l", "sa2alt", "sa2elt", "7kit m3a", "stannit", "chouft",
    "قلت ل", "قلت", "كلمت", "عيطت ل", "عيطت", "تلفنت ل", "تلفنت", "بعثت ل", "بعثت", "سألت", "حكيت مع",
    "شفت",  # `chouft` in Arabic script (review of #97)
)  # fmt: skip

# Verbs agreeing with a person, right after the name. `ja`/`jet`/`mcha`/`mchet` are kept knowing `sana jet`
# (the year came) gets masked.
VERBS_AFTER_NAME: tuple[str, ...] = (
    "9alli", "9al", "9alet", "kallamni", "3ayetli", "b3athli", "ja", "jet", "mcha", "mchet",
    "قالي", "قال", "قالت", "كلمني", "عيطلي", "بعثلي", "جا", "جات", "مشى", "مشات",
)  # fmt: skip

# Fixed expressions in which a common-word name is the ordinary word: (word before, the name, word after).
# Never a name unless a strong cue comes right before (`Mme Mabrouk !`). `Mabrouk` before `!` or `3lik` is
# handled with its spelling variants in the recognizer. Expressions with no listed name in them (`ya 7abibi`,
# `rabbi ysahel`, `inchallah`, `wa9t essala`) need no entry: their words aren't names.
NEVER_NAME_EXPRESSIONS: tuple[tuple[str, str, str], ...] = (
    ("alf", "mabrouk", ""), ("", "mabrouk", "el bac"), ("3id", "mabrouk", ""), ("3idek", "mabrouk", ""),
    ("ألف", "مبروك", ""), ("", "مبروك", "عليك"), ("عيد", "مبروك", ""),
    ("ramdhan", "karim", ""), ("romdhan", "karim", ""), ("ramadan", "karim", ""), ("rabbi", "karim", ""),
    ("allah", "karim", ""), ("رمضان", "كريم", ""), ("ربي", "كريم", ""), ("الله", "كريم", ""),
    ("", "sana", "sa3ida"), ("kol", "sana", ""), ("", "سنة", "سعيدة"), ("كل", "سنة", ""),
    ("3andi", "amal", ""), ("fama", "amal", ""), ("", "amal", "kbir"), ("عندي", "أمل", ""),
    ("عندي", "أمال", ""), ("", "رحمة", "الله"), ("", "رحمه", "الله"),
    ("", "nour", "3iniya"), ("", "نور", "عينيا"),
)  # fmt: skip

CITIES: tuple[str, ...] = (
    "Tunis", "Sfax", "Sousse", "Kairouan", "Bizerte", "Gabès", "Ariana", "Gafsa", "Monastir", "Ben Arous",
    "Kasserine", "Médenine", "Nabeul", "Tataouine", "Béja", "Le Kef", "Mahdia", "Sidi Bouzid", "Jendouba",
    "Tozeur", "Manouba", "Siliana", "Zaghouan", "Kébili", "La Marsa", "Carthage", "Hammamet", "Djerba",
    "Zarzis", "Houmt Souk", "La Goulette", "Sidi Bou Said", "El Menzah", "Ennasr", "Lac 2", "Raoued",
    "Mégrine", "Rades", "Hammam Lif", "Hammam Sousse", "Msaken", "Moknine", "Ksar Hellal", "Menzel Bourguiba",
    "El Kef", "Kelibia", "Korba", "Tabarka", "Douz", "Nefta", "Metlaoui",
    "Alger", "Oran", "Constantine", "Casablanca", "Rabat", "Marrakech", "Tripoli",
    "Paris", "Lyon", "Marseille", "Toulouse", "Nice", "Bordeaux", "Lille", "Nantes", "Strasbourg",
    "London", "Manchester",
    "تونس", "صفاقس", "سوسة", "القيروان", "بنزرت", "قابس", "أريانة", "قفصة", "المنستير", "نابل", "المهدية",
    "جربة", "الحمامات", "قرطاج", "المرسى",
)  # fmt: skip

# Cities that are also words or names: matched only with a capital letter (deep dive A).
COMMON_CITIES: tuple[str, ...] = ("Nice", "Douz", "Korba", "Ennasr", "Lac 2", "Carthage")

# Speech and contact verbs right after a name, which #50 uses as cues (`Hazem 9alli`, `Aymen kallamni`,
# `سامي قالي`); never names themselves. `ner.py` adds `said` and `a dit`: `said` is also a name (Saïd).
SPEECH_VERBS: tuple[str, ...] = (
    "9alli", "9al", "9alet", "9alt", "kallamni", "kallemni", "3ayetli", "3ayatli", "b3athli",
    "قالي", "قال", "قالت", "كلمني", "عيطلي", "بعثلي",
)  # fmt: skip

# Darija words that are never a name, in Latin letters and Arabic script (ADR-0008, condition 2). The NER
# model takes them into names or tags them alone (`nomrou Zakaria`, `Behi`, `Aymen 9alli`), so a model span is
# cut at them. Pronouns, function and question words, and frequent words of requests and replies. Arabizi
# words with digits need an entry too: a span is also cut at an unlisted one, but names are written with
# digits (`Mo3ez`), so that word is masked again on its own (`ner.py`); only a listed word stays in clear.
# French and English stopwords are the gazetteer's `STOPWORDS`. Words that are also given names are left out,
# since a model span is cut at these whatever its score: `wala` (Walaa), `bahi`, `rani`, `yesser`, `يسر`
# (Yosr), `ken`, `mel`, `win`, `ama`.
NEVER_NAMES: tuple[str, ...] = (
    # pronouns
    "ena", "enti", "enta", "inti", "inta", "houwa", "hiya", "ahna", "a7na", "entouma", "homa",
    "انا", "أنا", "انت", "إنت", "انتي", "إنتي", "احنا", "نحن", "انتوما", "هوما", "هم",
    # function and question words
    "w", "wa", "ma", "mech", "mich", "fi", "f", "l", "b", "men", "m3a", "ma3", "3al", "3and", "3nd", "3andi",
    "5ater", "5atr",
    "mta3", "mte3", "chnowa", "chnouwa", "chneya", "chkoun", "kifech", "kifesh", "wa9tech", "9adech",
    "متاع", "متاعو", "متاعها", "متاعي", "نتاع", "شنوة", "شنية", "شكون", "كيفاش", "وقتاش", "وين", "قداش",
    # time
    "taw", "tawa", "lyoum", "ghodwa", "ba3d", "9bal", "lbare7", "sbe7", "3asr",
    "توا", "اليوم", "غدوة", "بعد", "قبل",
    # replies, thanks, blessings
    "behi", "barcha", "yezzi", "zeda", "zada", "ey", "eyh", "iyeh", "aslema", "ahla",
    "marhba", "rabbi", "inchallah", "nchallah", "hamdoulah", "sahit", "y3aychek", "ya3tik", "y5allik",
    "3aslema",
    "باهي", "برشا", "يزي", "زادة", "إيه", "اي", "اهلا", "مرحبا", "ربي", "إنشاء", "يعيشك", "يعطيك",
    # people and places of everyday life
    "rajel", "mra", "راجل", "مرا", "5ouya", "5outi", "3ayla", "7ouma", "9ahwa",
    # things people ask for
    "nomrou", "noumrou", "numrou", "nmrou", "rendez", "rdv", "mail", "contrat", "flous",
    "نومرو", "النومرو", "رقم", "الرقم", "موعد", "العقد", "فلوس",
    # speech, contact and movement verbs: #50's cues (`SPEECH_VERBS`), except `kallemni` and `بعثلي`, which
    # have no digits, so listing them would change where spans are cut (`ner.py`)
    "kallem", "kallamt", "kallemt", "kallamni", "9alli", "9al", "9alet", "9alt", "3ayetli", "3ayatli",
    "b3athli", "ja", "jet", "mcha", "mchet",
    "قالي", "قال", "قالت", "كلم", "كلمني", "عيطلي", "جا", "جات", "مشى", "مشات",
    # sending and doing, in requests (`ab3athli el mail`, `3malt rdv`)
    "ab3ath", "ab3athli", "ab3athlou", "ab3athtlou", "ab3atht", "b3atht", "3malt", "3mel", "3melt",
    # frequent verbs the model takes into a name (`Sami y7eb Ali`, `Mootaz ya5dem Sfax`)
    "y7eb", "n7eb", "t7eb", "ya5dem", "na5dem", "ta5dem", "y7ki", "n7ki", "t7ki",
)  # fmt: skip
