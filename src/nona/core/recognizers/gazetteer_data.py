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

# Darija words that are never a name, in Latin letters and Arabic script (ADR-0008, condition 2). The NER
# model takes them into names or tags them alone (`nomrou Zakaria`, `Behi`, `Aymen 9alli`), so a model span is
# cut at them. Pronouns, function and question words, and frequent words of requests and replies. Arabizi
# words with digits (`9alli`, `y3aychek`) need no entry: an unlisted one is never a name (`ner.py`). French
# and English stopwords are the gazetteer's `STOPWORDS`.
NEVER_NAMES: tuple[str, ...] = (
    # pronouns
    "ena", "enti", "enta", "inti", "inta", "houwa", "hiya", "ahna", "a7na", "entouma", "homa", "rani", "rahi",
    "انا", "أنا", "انت", "إنت", "انتي", "إنتي", "احنا", "نحن", "انتوما", "هوما", "هم",
    # function and question words
    "w", "wa", "wala", "walla", "ama", "ken", "kan", "ma", "mech", "mich", "fi", "f", "l", "b", "men", "mel",
    "mta3", "mte3", "chnowa", "chnouwa", "chneya", "chkoun", "kifech", "kifesh", "win", "wa9tech", "9adech",
    "متاع", "متاعو", "متاعها", "متاعي", "نتاع", "شنوة", "شنية", "شكون", "كيفاش", "وقتاش", "وين", "قداش",
    # time
    "taw", "tawa", "lyoum", "ghodwa", "ba3d", "9bal", "lbare7", "توا", "اليوم", "غدوة", "بعد", "قبل",
    # replies, thanks, blessings
    "behi", "bahi", "barcha", "yesser", "yezzi", "zeda", "zada", "ey", "eyh", "iyeh", "aslema", "ahla",
    "marhba", "rabbi", "inchallah", "nchallah", "hamdoulah", "sahit", "y3aychek", "ya3tik",
    "باهي", "برشا", "يسر", "يزي", "زادة", "إيه", "اي", "اهلا", "مرحبا", "ربي", "إنشاء", "يعيشك", "يعطيك",
    # people
    "rajel", "mra", "راجل", "مرا",
    # things people ask for
    "nomrou", "noumrou", "numrou", "nmrou", "rendez", "rdv", "mail", "contrat", "flous",
    "نومرو", "النومرو", "رقم", "الرقم", "موعد", "العقد", "فلوس",
    # speech and contact verbs, which #50 uses as cues around a name
    "kallem", "kallamt", "kallemt", "kallamni", "9alli", "9al", "9alet", "ja", "jet", "mcha", "mchet",
    "قالي", "قال", "قالت", "كلم", "كلمني", "عيطلي", "جا", "جات", "مشى", "مشات",
)  # fmt: skip
