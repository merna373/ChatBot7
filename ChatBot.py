import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import re
import requests


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not HF_TOKEN:
    st.error("HF_TOKEN is missing from the .env file.")
    st.stop()

if not SERPER_API_KEY:
    st.warning(
        "SERPER_API_KEY is missing from the .env file. "
        "Web search will not be available."
    )

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

MODEL_NAME = "openai/gpt-oss-20b"

st.set_page_config(
    page_title="Kemet - Discover Egypt",
    page_icon="🏺",
    layout="wide"
)


# =========================================================
# LOAD KNOWLEDGE BASE
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(BASE_DIR, "knowledge_base.json")


@st.cache_data
def load_knowledge_base():

    if not os.path.exists(KB_PATH):

        st.error(
            f"❌ Knowledge Base file was not found.\n\n"
            f"Expected location:\n{KB_PATH}"
        )

        return {
            "places": [],
            "shopping": [],
            "food": [],
            "transport": [],
            "tickets": [],
            "photography": [],
            "culture_etiquette": [],
            "weather": [],
            "safety": [],
            "trip_planning": [],
            "general_faq": [],
            "general_rules": []
        }

    try:

        with open(KB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    except json.JSONDecodeError as e:

        st.error(
            f"❌ knowledge_base.json contains invalid JSON.\n\n"
            f"{e}"
        )

        return {
            "places": [],
            "shopping": [],
            "food": [],
            "transport": [],
            "tickets": [],
            "photography": [],
            "culture_etiquette": [],
            "weather": [],
            "safety": [],
            "trip_planning": [],
            "general_faq": [],
            "general_rules": []
        }


kb = load_knowledge_base()


# =========================================================
# KNOWLEDGE BASE SECTIONS
# =========================================================

KB_SECTIONS = [
    "places",
    "shopping",
    "food",
    "transport",
    "tickets",
    "photography",
    "culture_etiquette",
    "weather",
    "safety",
    "trip_planning",
    "general_faq"
]


total_kb_items = sum(
    len(kb.get(section, []))
    for section in KB_SECTIONS
    if isinstance(kb.get(section, []), list)
)

places = kb.get("places", [])
general_rules = kb.get("general_rules", [])


# =========================================================
# LANGUAGE DETECTION
# =========================================================
def detect_language(text):
    if not text:
        return "en"

    text_lower = text.lower().strip()

    # Arabic
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"

    # Russian / Cyrillic
    if re.search(r"[\u0400-\u04FF]", text):
        return "ru"

    # German
    german_words = {
        "der", "die", "das", "und", "ich", "ist",
        "wie", "wo", "was", "für", "mit",
        "möchte", "möchten", "reise", "reisen",
        "ägypten", "ägyptische", "ägyptischen",
        "kann", "gibt", "tourist",
        "sehenswürdigkeiten", "sind",
        "welche", "was", "nach"
    }

    words = set(
        re.findall(
            r"\b[\wäöüßÄÖÜ]+\b",
            text_lower
        )
    )

    german_matches = words.intersection(german_words)

    if len(german_matches) >= 1:
        return "de"

    # Default = English
    return "en"


# =========================================================
# ERROR MESSAGES
# =========================================================

ERROR_MESSAGES = {

    "ar":
        "❌ حدث خطأ أثناء الاتصال بالنموذج. حاول مرة أخرى.",

    "en":
        "❌ An error occurred while connecting to the AI model. Please try again.",

    "ru":
        "❌ Произошла ошибка при подключении к модели ИИ. Попробуйте ещё раз.",

    "de":
        "❌ Beim Verbinden mit dem KI-Modell ist ein Fehler aufgetreten. Bitte versuchen Sie es erneut."
}


# =========================================================
# UI TEXT
# =========================================================

UI_TEXT = {

    "ar": {

        "title":
            "🏺 كيميت – اكتشف مصر من خلال المحادثة",

        "welcome":
            (
                "أهلاً بك في كيميت 🇪🇬\n\n"
                "اسألني عن الأماكن السياحية، الطعام، التسوق، "
                "المواصلات، الثقافة، الطقس، الأمان، والتخطيط للرحلات في مصر."
            ),

        "knowledge":
            "قاعدة المعرفة",

        "items":
            "معلومة",

        "clear":
            "مسح المحادثة",

        "input":
            "اكتب سؤالك هنا...",

        "not_found":
            (
                "لم أجد معلومات موثوقة كافية عن هذا السؤال "
                "في قاعدة المعرفة أو البحث المتاح."
            )
    },

    "en": {

        "title":
            "🏺 Kemet – Discover Egypt Through Conversation",

        "welcome":
            (
                "Welcome to Kemet 🇪🇬\n\n"
                "Ask me about tourist attractions, food, shopping, "
                "transportation, culture, weather, safety, and trip planning in Egypt."
            ),

        "knowledge":
            "Knowledge Base",

        "items":
            "items",

        "clear":
            "Clear Chat",

        "input":
            "Ask something about Egypt...",

        "not_found":
            (
                "I don't have enough verified information "
                "about this topic in the Knowledge Base or available web results."
            )
    },

    "ru": {

        "title":
            "🏺 Кемет — откройте Египет через общение",

        "welcome":
            (
                "Добро пожаловать в Кемет 🇪🇬\n\n"
                "Спросите меня о достопримечательностях, еде, покупках, "
                "транспорте, культуре, погоде, безопасности и планировании поездки в Египет."
            ),

        "knowledge":
            "База знаний",

        "items":
            "записей",

        "clear":
            "Очистить чат",

        "input":
            "Задайте вопрос о Египте...",

        "not_found":
            (
                "В моей базе знаний или доступных веб-результатах "
                "недостаточно проверенной информации."
            )
    },

    "de": {

        "title":
            "🏺 Kemet – Ägypten im Gespräch entdecken",

        "welcome":
            (
                "Willkommen bei Kemet 🇪🇬\n\n"
                "Fragen Sie mich nach Sehenswürdigkeiten, Essen, Shopping, "
                "Transport, Kultur, Wetter, Sicherheit und Reiseplanung in Ägypten."
            ),

        "knowledge":
            "Wissensdatenbank",

        "items":
            "Einträge",

        "clear":
            "Chat löschen",

        "input":
            "Stellen Sie eine Frage über Ägypten...",

        "not_found":
            (
                "Ich habe zu diesem Thema nicht genügend "
                "verifizierte Informationen in meiner Wissensdatenbank "
                "oder in den verfügbaren Web-Ergebnissen."
            )
    }
}


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(text):

    if not text:
        return ""

    text = str(text).lower().strip()

    # Arabic normalization
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")

    # Arabic diacritics
    text = re.sub(
        r"[\u064B-\u065F\u0670]",
        "",
        text
    )

    # Remove punctuation
    text = re.sub(
        r"[^\w\s\u0600-\u06FF\u0400-\u04FF]",
        " ",
        text
    )

    # Extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# EXTRACT TEXT FROM KB ITEM
# =========================================================

def extract_item_text(item):

    text_parts = []

    # ID / category
    for key in [
        "id",
        "subcategory",
        "category"
    ]:

        value = item.get(key, "")

        if value:
            text_parts.append(str(value))

    # Names
    name = item.get("name", "")

    if isinstance(name, dict):

        for value in name.values():

            if value:
                text_parts.append(str(value))

    elif name:

        text_parts.append(str(name))

    # Old format compatibility
    for key in [
        "name_ar",
        "name_en",
        "name_ru",
        "name_de"
    ]:

        value = item.get(key, "")

        if value:
            text_parts.append(str(value))

    # Questions
    questions = item.get("questions", {})

    if isinstance(questions, dict):

        for values in questions.values():

            if isinstance(values, list):

                text_parts.extend(
                    str(value)
                    for value in values
                    if value
                )

            elif values:

                text_parts.append(str(values))

    elif isinstance(questions, list):

        text_parts.extend(
            str(value)
            for value in questions
            if value
        )

    # Keywords
    keywords = item.get("keywords", [])

    if isinstance(keywords, list):

        text_parts.extend(
            str(value)
            for value in keywords
            if value
        )

    elif keywords:

        text_parts.append(str(keywords))

    # Aliases
    aliases = item.get("aliases", {})

    if isinstance(aliases, dict):

        for language_aliases in aliases.values():

            if isinstance(language_aliases, list):

                text_parts.extend(
                    str(value)
                    for value in language_aliases
                    if value
                )

            elif language_aliases:

                text_parts.append(
                    str(language_aliases)
                )

    # Descriptions
    for key in [
        "description",
        "description_ar",
        "description_en",
        "description_ru",
        "description_de"
    ]:

        value = item.get(key, "")

        if isinstance(value, dict):

            for v in value.values():

                if v:
                    text_parts.append(str(v))

        elif value:

            text_parts.append(str(value))

    # Tags
    tags = item.get("tags", [])

    if isinstance(tags, list):

        text_parts.extend(
            str(value)
            for value in tags
            if value
        )

    # Activities
    activities = item.get("activities", [])

    if isinstance(activities, list):

        text_parts.extend(
            str(value)
            for value in activities
            if value
        )

    # Linked places
    linked_places = item.get(
        "linked_places",
        []
    )

    if isinstance(linked_places, list):

        text_parts.extend(
            str(value)
            for value in linked_places
            if value
        )

    # Location
    for key in [
        "governorate",
        "governorate_ar",
        "city",
        "city_ar",
        "area"
    ]:

        value = item.get(key, "")

        if value:
            text_parts.append(str(value))

    return " ".join(text_parts)


# =========================================================
# SEARCH KNOWLEDGE BASE
# =========================================================

def search_knowledge_base(query, max_results=6):

    query_normalized = normalize_text(query)

    if not query_normalized:
        return []

    query_words = set(query_normalized.split())

    # -----------------------------------------------------
    # Detect the most likely section from the user's query
    # -----------------------------------------------------

    section_keywords = {
        "food": [
            "food", "foods", "dish", "dishes", "meal", "meals",
            "eat", "eating", "cuisine", "egyptian food",
            "أكل", "اكل", "أكلات", "اكلات", "طعام",
            "أكلة", "اكلة", "مأكولات", "ماكولات",
            "وجبة", "وجبات", "مطبخ", "المطبخ",
            "кошари", "еда", "блюдо",
            "essen", "gerichte", "speisen",
            "ägyptisches essen"
        ],

        "places": [
            "place", "places", "attraction", "attractions",
            "tourist", "tourism", "visit", "visiting",
            "landmark", "landmarks", "sightseeing",
            "مكان", "اماكن", "أماكن", "سياحة",
            "سياحية", "سياحي", "معالم", "مزارات",
            "достопримечательности", "туризм",
            "sehenswürdigkeiten", "tourismus"
        ],

        "shopping": [
            "shopping", "shop", "shops", "market", "markets",
            "souvenir", "souvenirs", "gift", "gifts",
            "تسوق", "تسوق", "سوق", "أسواق", "اسواق",
            "هدايا", "هدية", "هدايا تذكارية",
            "покупки", "рынок", "подарки",
            "einkaufen", "markt", "geschenke"
        ],

        "transport": [
            "transport", "transportation", "taxi", "train",
            "trains", "flight", "flights", "bus", "metro",
            "travel", "get to", "how to reach",
            "مواصلات", "المواصلات", "قطار", "قطارات",
            "طيران", "طيارة", "حافلة", "اتوبيس", "أتوبيس",
            "مترو", "تاكسي", "ازاي اوصل", "كيف اصل",
            "как добраться", "транспорт", "поезд",
            "самолет", "метро", "taxi",
            "verkehrsmittel", "zug", "flugzeug", "metro"
        ],

        "weather": [
            "weather", "forecast", "temperature",
            "الطقس", "الجو", "درجة الحرارة", "درجه الحراره",
            "погода", "температура",
            "wetter", "temperatur"
        ],

        "safety": [
            "safe", "safety", "danger", "dangerous",
            "أمان", "امان", "سلامة", "سلامه", "خطر",
            "безопасность", "опасно",
            "sicherheit", "sicher", "gefährlich"
        ],

        "culture_etiquette": [
            "culture", "customs", "traditions", "etiquette",
            "ثقافة", "ثقافه", "عادات", "تقاليد", "اتيكيت",
            "культура", "традиции",
            "kultur", "traditionen", "etikette"
        ],

        "trip_planning": [
            "trip", "plan", "planning", "itinerary",
            "رحلة", "رحله", "تخطيط", "برنامج", "خط سير",
            "поездка", "план поездки",
            "reise", "reiseplanung", "reiseplan"
        ],

        "tickets": [
            "ticket", "tickets", "price", "prices",
            "تذكرة", "تذاكر", "سعر", "اسعار", "أسعار",
            "билет", "билеты", "цена",
            "ticket", "preis", "preise"
        ]
    }

    # -----------------------------------------------------
    # Find relevant sections
    # -----------------------------------------------------

    detected_sections = []

    for section, keywords in section_keywords.items():

        for keyword in keywords:

            keyword_normalized = normalize_text(keyword)

            if not keyword_normalized:
                continue

            if keyword_normalized in query_normalized:
                detected_sections.append(section)
                break

    # -----------------------------------------------------
    # Search KB
    # -----------------------------------------------------

    results = []

    for section in KB_SECTIONS:

        items = kb.get(section, [])

        if not isinstance(items, list):
            continue

        # Give a strong boost to the detected section
        section_boost = 0

        if section in detected_sections:
            section_boost = 20

        for item in items:

            if not isinstance(item, dict):
                continue

            score = section_boost

            searchable_text = extract_item_text(item)

            searchable_normalized = normalize_text(
                searchable_text
            )

            if not searchable_normalized:
                continue

            searchable_words = set(
                searchable_normalized.split()
            )

            # -------------------------------------------------
            # Exact phrase
            # -------------------------------------------------

            if query_normalized in searchable_normalized:
                score += 15

            # -------------------------------------------------
            # Word matching
            # -------------------------------------------------

            matching_words = query_words.intersection(
                searchable_words
            )

            score += len(matching_words) * 2

            # -------------------------------------------------
            # ID
            # -------------------------------------------------

            item_id = normalize_text(
                str(item.get("id", ""))
            )

            if item_id:

                id_words = set(
                    item_id.replace("_", " ").split()
                )

                score += len(
                    query_words.intersection(id_words)
                ) * 3

            # -------------------------------------------------
            # Subcategory
            # -------------------------------------------------

            subcategory = normalize_text(
                str(item.get("subcategory", ""))
            )

            if subcategory:

                subcategory_words = set(
                    subcategory.replace("_", " ").split()
                )

                score += len(
                    query_words.intersection(
                        subcategory_words
                    )
                ) * 4

            # -------------------------------------------------
            # Keywords
            # -------------------------------------------------

            keywords = item.get("keywords", [])

            if isinstance(keywords, list):

                for keyword in keywords:

                    keyword_normalized = normalize_text(
                        keyword
                    )

                    if not keyword_normalized:
                        continue

                    if keyword_normalized in query_normalized:
                        score += 8

                    keyword_words = set(
                        keyword_normalized.split()
                    )

                    score += len(
                        query_words.intersection(
                            keyword_words
                        )
                    ) * 3

            # -------------------------------------------------
            # Questions
            # -------------------------------------------------

            questions = item.get("questions", {})

            if isinstance(questions, dict):

                question_values = []

                for values in questions.values():

                    if isinstance(values, list):
                        question_values.extend(values)

                    elif values:
                        question_values.append(values)

                for question in question_values:

                    question_normalized = normalize_text(
                        question
                    )

                    if not question_normalized:
                        continue

                    if query_normalized == question_normalized:
                        score += 20

                    elif query_normalized in question_normalized:
                        score += 10

            # -------------------------------------------------
            # Add result
            # -------------------------------------------------

            if score > 0:

                results.append(
                    (
                        score,
                        section,
                        item
                    )
                )

    # ---------------------------------------------------------
    # Special handling for general food questions
    # ---------------------------------------------------------
    
    # If the user asks generally about Egyptian food,
    # return multiple food items instead of only one item.

    if "food" in detected_sections:

        food_results = [
            result
            for result in results
            if result[1] == "food"
        ]

        food_results.sort(
            key=lambda x: x[0],
            reverse=True
        )

        if food_results:

            return food_results[:max_results]

    # ---------------------------------------------------------
    # Final sorting
    # ---------------------------------------------------------

    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return results[:max_results]


# =========================================================
# FORMAT KNOWLEDGE BASE
# =========================================================

def get_multilingual_value(
    value,
    language
):

    if isinstance(value, dict):

        if value.get(language):
            return value.get(language)

        if value.get("en"):
            return value.get("en")

        if value.get("ar"):
            return value.get("ar")

        return ""

    return str(value) if value else ""


def format_knowledge(
    results,
    language
):

    if not results:
        return "NO VERIFIED KNOWLEDGE WAS FOUND."

    text_parts = []

    for score, section, item in results:

        name = get_multilingual_value(
            item.get("name", ""),
            language
        )

        if not name:

            name_key = f"name_{language}"

            name = item.get(
                name_key,
                item.get("name_en", "")
            )

        answer = get_multilingual_value(
            item.get("answer", ""),
            language
        )

        description = get_multilingual_value(
            item.get("description", ""),
            language
        )

        if not description:

            description_key = (
                f"description_{language}"
            )

            description = item.get(
                description_key,
                ""
            )

        section_name = section.replace(
            "_",
            " "
        ).title()

        subcategory = item.get(
            "subcategory",
            ""
        )

        chunk = (
            f"--- VERIFIED KNOWLEDGE ---\n"
            f"Section: {section_name}\n"
            f"Subcategory: {subcategory}\n"
            f"Name: {name}\n"
        )

        if answer:

            chunk += (
                f"Verified Answer: {answer}\n"
            )

        if description:

            chunk += (
                f"Verified Description: {description}\n"
            )

        activities = item.get(
            "activities",
            []
        )

        if isinstance(activities, list) and activities:

            chunk += (
                "Activities: "
                + ", ".join(
                    str(x)
                    for x in activities
                )
                + "\n"
            )

        linked_places = item.get(
            "linked_places",
            []
        )

        if isinstance(
            linked_places,
            list
        ) and linked_places:

            chunk += (
                "Linked Places: "
                + ", ".join(
                    str(x)
                    for x in linked_places
                )
                + "\n"
            )

        source = item.get(
            "source",
            ""
        )

        if source:

            chunk += (
                f"Source: {source}\n"
            )

        text_parts.append(chunk)

    return "\n".join(text_parts)


# =========================================================
# WEB SEARCH DECISION
# =========================================================

def needs_web_search(query, knowledge_results):
    query_normalized = normalize_text(query)

    # Current / dynamic information
    dynamic_keywords = [
        "today", "now", "current", "currently",
        "price", "prices", "cost", "how much",
        "opening hours", "open now", "schedule",
        "availability", "booking", "event", "weather",
        "today's", "tomorrow",

        "دلوقتي", "الان", "الآن", "حاليا", "حاليًا",
        "سعر", "اسعار", "أسعار", "بكام", "بكم",
        "تكلفه", "تكلفة", "مواعيد", "مفتوح",
        "النهارده", "اليوم", "بكره", "غدًا",
        "حجز", "متاح", "طقس",

        "цена", "цены", "сейчас", "сегодня",
        "стоимость", "расписание",

        "preis", "preise", "kosten", "heute",
        "öffnungszeiten", "verfügbarkeit"
    ]

    for keyword in dynamic_keywords:
        if normalize_text(keyword) in query_normalized:
            return True

    # Specific tourism places / businesses
    specific_place_keywords = [
        "market", "markets",
        "souvenir", "souvenirs",
        "gift", "gifts",
        "restaurant", "restaurants",
        "hotel", "hotels",
        "shop", "shops",
        "store", "stores",

        "هدايا", "هدايا تذكارية",
        "سوق", "اسواق", "أسواق",
        "محل", "محلات",
        "مطعم", "مطاعم",
        "فندق", "فنادق",

        "рынок", "подарок", "подарки",
        "ресторан", "отель",

        "markt", "geschenke",
        "restaurant", "hotel"
    ]

    for keyword in specific_place_keywords:
        if normalize_text(keyword) in query_normalized:
            return True

    # If there is no relevant knowledge, search the web
    if not knowledge_results:
        return True

    # Otherwise use the Knowledge Base when it strongly matches
    best_score = knowledge_results[0][0]

    if best_score < 8:
        return True

    return False

# =========================================================
# WEB SEARCH
# =========================================================

def search_web(
    query,
    language="en",
    num_results=5
):

    if not SERPER_API_KEY:
        return []

    language_map = {
        "ar": "ar",
        "en": "en",
        "de": "de",
        "ru": "ru"
    }

    hl = language_map.get(
        language,
        "en"
    )

    url = "https://google.serper.dev/search"

    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "q": f"{query} Egypt",
        "gl": "eg",
        "hl": hl,
        "num": num_results
    }

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        results = []

        for item in data.get("organic", []):

            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", "")
            })

        return results

    except Exception as e:

        print("Web search error:", e)

        return []


# =========================================================
# FORMAT WEB RESULTS
# =========================================================

def format_web_results(
    results
):

    if not results:

        return "NO WEB RESULTS WERE FOUND."

    text_parts = []

    for index, result in enumerate(
        results,
        start=1
    ):

        title = result.get(
            "title",
            ""
        )

        snippet = result.get(
            "snippet",
            ""
        )

        link = result.get(
            "link",
            ""
        )

        text_parts.append(
            f"--- WEB RESULT {index} ---\n"
            f"Title: {title}\n"
            f"Summary: {snippet}\n"
            f"URL: {link}\n"
        )

    return "\n".join(text_parts)


# =========================================================
# SYSTEM PROMPT
# =========================================================

def build_system_prompt(
    language,
    knowledge,
    web_results=""
):

    language_names = {

        "ar": "Arabic",
        "en": "English",
        "ru": "Russian",
        "de": "German"
    }

    language_name = language_names.get(
        language,
        "English"
    )

    return f"""

You are Kemet – Discover Egypt Through Conversation.

You are an Egyptian tourism assistant.

=========================================================
STRICT LANGUAGE RULE
=========================================================

The user's LATEST message is the ONLY source for determining
the response language.

The application has EXACTLY FOUR supported languages:

- en = English
- ar = Arabic
- ru = Russian
- de = German

Detected language for the latest user message:
{language_name}

Detected language code:
{language}

The response MUST be written 100% in the detected language.

ABSOLUTE RULES:

- If language = "en", write ONLY in English.
- If language = "ar", write ONLY in Arabic.
- If language = "ru", write ONLY in Russian.
- If language = "de", write ONLY in German.

NEVER mix languages.

The Knowledge Base may contain information in multiple languages.
Ignore the language of the Knowledge Base when choosing the
response language.

Previous assistant messages may be written in another language.
IGNORE their language completely.

Do NOT copy the language, style, or wording of previous assistant
answers.

Always answer according to the language of the LATEST user message.

Before producing the final answer, silently check:

1. What is the detected language code?
2. Is every sentence in that language?
3. Did I accidentally use another language?

If any sentence is in another language, rewrite it before answering.

Examples:

User:
"What are the most famous tourist attractions in Egypt?"

Detected language: en

Correct response language:
English ONLY.

User:
"ما هي أشهر الأماكن السياحية في مصر؟"

Detected language: ar

Correct response language:
Arabic ONLY.

User:
"Какие самые известные туристические достопримечательности Египта?"

Detected language: ru

Correct response language:
Russian ONLY.

User:
"Was sind die berühmtesten Sehenswürdigkeiten in Ägypten?"

Detected language: de

Correct response language:
German ONLY.

=========================================================
KNOWLEDGE SOURCE PRIORITY
=========================================================

You have two information sources:

1. VERIFIED KNOWLEDGE BASE
2. WEB SEARCH RESULTS

Use the Knowledge Base first whenever it directly answers
the user's question.

If the Knowledge Base does not contain enough relevant
information, use the supplied Web Search Results.

If the question asks for current or dynamic information,
Web Search Results should be preferred.

Examples of dynamic information:

- current ticket prices
- today's opening hours
- current events
- current weather
- tomorrow's weather
- current transportation schedules
- current train schedules
- current flight information
- current restaurant information
- current availability
- current booking information

=========================================================
ANTI-HALLUCINATION RULE
=========================================================

NEVER invent information.

If information is not supported by either:

- the Knowledge Base
OR
- the Web Search Results

say clearly that verified information is not available.

Do not use your internal memory to invent missing facts.

=========================================================
CHATBOT SCOPE
=========================================================

Kemet is a tourism assistant specialized in Egypt.

The chatbot should primarily answer questions related to:

- Tourist attractions and historical places in Egypt
- Food and Egyptian cuisine
- Shopping and souvenirs
- Transportation
- Tickets and visiting information
- Photography
- Egyptian culture and etiquette
- Weather and safety for travelers
- Trip planning and tourism-related questions

If the user asks a question that is completely unrelated to
tourism in Egypt, do not try to force the question into a
tourism context.

For unrelated questions, politely explain that Kemet is
specialized in tourism in Egypt and invite the user to ask
about Egypt tourism.

For example, if the user asks:
"5*5 بكام؟"

Respond in the detected language with a short message explaining
that Kemet is an Egyptian tourism assistant and can help with
tourist attractions, food, shopping, transportation, culture,
weather, safety, and trip planning.

Do not provide detailed answers to unrelated topics.

=========================================================
WEB SEARCH RULES
=========================================================

When Web Search Results are provided:

- Use them as evidence.
- Prefer official websites and authoritative sources.
- Do not treat search snippets as perfect or guaranteed.
- If different sources disagree, mention the uncertainty.
- Do not invent a fact that is not supported by the results.
- Do not create fake URLs.
- Do not claim that a source is official unless the source
  clearly appears to be an official website.

=========================================================
IMPORTANT
=========================================================

Never invent:

- tourist attractions
- museums
- restaurants
- cafes
- hotels
- markets
- shops
- streets
- transportation routes
- bus numbers
- metro stations
- ticket prices
- ticket policies
- opening hours
- events
- addresses
- phone numbers
- websites
- current availability
- historical claims
- unsupported statistics

Do not claim something is:

- the biggest
- the oldest
- the longest
- the most famous
- the best

unless supported by the provided information.

=========================================================
STRICT EVIDENCE RULE
=========================================================

For every specific factual claim, use ONLY information explicitly
supported by the provided Knowledge Base or Web Search Results.

This includes:
- exact numbers
- prices
- price ranges
- number of shops
- opening and closing hours
- addresses
- distances
- transportation routes
- names of markets, shops, restaurants, hotels, museums, or attractions
- events and activities
- historical dates and claims

If a specific fact is not explicitly supported by the provided
Knowledge Base or Web Search Results:
- DO NOT guess
- DO NOT estimate
- DO NOT infer
- DO NOT use typical or remembered values

=========================================================
PRICE RULE
=========================================================

For questions about prices, costs, budgets, or how much something
costs:

- Give a price ONLY if that price is explicitly supported by the
  provided Knowledge Base or Web Search Results.
- Do NOT generate approximate prices from internal knowledge.
- Do NOT create price ranges such as "50-200 EGP" unless the sources
  explicitly support that range.
- If verified current prices are unavailable, clearly say that
  current verified prices are not available.
- You may suggest souvenir categories without giving unsupported
  prices.

=========================================================
PLACE NAME RULE
=========================================================

Mention a specific:
- market
- shop
- restaurant
- hotel
- museum
- tourist attraction
- street
- transportation station

ONLY if its exact name appears in the provided Knowledge Base or
Web Search Results.

Do NOT turn:
- a city
- a neighborhood
- a generic category
- a tourist area

into the name of a specific place.

For example, do not invent names such as:
- "Souk Luxor"
- "Sharm El Sheikh Market"
- "Cairo Market"

unless that exact place name is supported by the provided sources.

=========================================================
CURRENT INFORMATION RULE
=========================================================

For current or dynamic information such as:
- today's prices
- opening hours
- current events
- availability
- booking
- weather
- transportation schedules

use Web Search Results when they are provided.

If the available sources do not contain the requested current
information, say that verified current information is unavailable.

=========================================================
SOURCE ATTRIBUTION RULE
=========================================================

If Web Search Results are used, do not claim that the information
came from the Knowledge Base unless it is actually supported by the
Knowledge Base.

Do not write a generic statement such as:
"المصادر من قاعدة المعرفة الموثقة"

unless the answer is actually supported by the Knowledge Base.

Do not invent or fabricate sources or URLs.

=========================================================
ANSWER STYLE
=========================================================

Keep answers:

- helpful
- concise
- clear
- natural
- easy to understand

Use bullet points when useful.

Do not overwhelm the user.

=========================================================
VERIFIED KNOWLEDGE BASE
=========================================================

{knowledge}

=========================================================
WEB SEARCH RESULTS
=========================================================

{web_results}

=========================================================
FINAL RESPONSE RULE
=========================================================
Answer the user's question directly.
Do not explain the internal architecture of Kemet.
Do not mention "Knowledge Base" or "Web Search Results"
unless it is useful for explaining the source of information.

Do not add a Sources section to the answer.
Sources are displayed separately by the application UI.
"""


# =========================================================
# CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": UI_TEXT["en"]["welcome"]
        }
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## ⚙️ Settings")

    detected_sidebar_language = "en"

    if st.session_state.messages:

        for message in reversed(
            st.session_state.messages
        ):

            if message["role"] == "user":

                detected_sidebar_language = detect_language(
                    message["content"]
                )

                break

    ui = UI_TEXT.get(
        detected_sidebar_language,
        UI_TEXT["en"]
    )

    # Knowledge Base count
    st.write(
        f"📚 {ui['knowledge']}: "
        f"{total_kb_items} {ui['items']}"
    )

    # Web Search status
    if SERPER_API_KEY:

        st.success(
            "🌐 Web Search: Enabled"
        )

    else:

        st.warning(
            "🌐 Web Search: Disabled"
        )

    # Sections
    with st.expander("📂 Knowledge Base Sections"):

        for section in KB_SECTIONS:

            count = len(
                kb.get(section, [])
            )

            display_name = section.replace(
                "_",
                " "
            ).title()

            st.write(
                f"• {display_name}: {count}"
            )

    # Clear chat
    if st.button(
        f"🗑️ {ui['clear']}",
        use_container_width=True
    ):

        st.session_state.messages = [

            {
                "role": "assistant",
                "content": ui["welcome"]
            }

        ]

        st.rerun()


# =========================================================
# TITLE
# =========================================================

current_language = "en"

for message in reversed(
    st.session_state.messages
):

    if message["role"] == "user":

        current_language = detect_language(
            message["content"]
        )

        break


ui = UI_TEXT.get(
    current_language,
    UI_TEXT["en"]
)

st.title(
    ui["title"]
)


# =========================================================
# DISPLAY CHAT
# =========================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources saved with this message
        if message.get("web_results"):
            with st.expander("🌐 Sources"):
                for result in message["web_results"]:
                    title = result.get("title", "Source")
                    link = result.get("link", "")
                    snippet = result.get("snippet", "")

                    st.markdown(f"**{title}**")

                    if snippet:
                        st.write(snippet)

                    if link:
                        st.markdown(f"[🔗 Open source]({link})")


# =========================================================
# CHAT INPUT
# =========================================================

user_prompt = st.chat_input(
    ui["input"]
)


# =========================================================
# PROCESS USER MESSAGE
# =========================================================

if user_prompt:

    # -----------------------------------------------------
    # Detect language
    # -----------------------------------------------------

    language = detect_language(
        user_prompt
    )

    # -----------------------------------------------------
    # Save user message
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )

    with st.chat_message("user"):

        st.markdown(
            user_prompt
        )

    # -----------------------------------------------------
    # Search Knowledge Base
    # -----------------------------------------------------

    knowledge_results = search_knowledge_base(
        user_prompt,
        max_results=6
    )

    # -----------------------------------------------------
    # Format Knowledge Base
    # -----------------------------------------------------

    knowledge_text = format_knowledge(
        knowledge_results,
        language
    )

    # -----------------------------------------------------
    # Decide whether Web Search is needed
    # -----------------------------------------------------

    use_web_search = needs_web_search(
        user_prompt,
        knowledge_results
    )

    web_results = []

    if use_web_search:

        with st.spinner("🔎 Searching the web..."):

            web_results = search_web(
                user_prompt,
                language,
                num_results=5
            )

    # -----------------------------------------------------
    # Format Web Results
    # -----------------------------------------------------

    web_text = format_web_results(
        web_results
    )

    # -----------------------------------------------------
    # Build System Prompt
    # -----------------------------------------------------

    system_prompt = build_system_prompt(
        language,
        knowledge_text,
        web_text
    )

    # -----------------------------------------------------
    # Keep recent conversation context
    # -----------------------------------------------------

    recent_messages = []

    for message in st.session_state.messages[-10:]:
        if message["role"] == "user":
            recent_messages.append(message)

        elif message["role"] == "assistant":
            if detect_language(message["content"]) == language:
                recent_messages.append(message)

    messages = [

        {
            "role": "system",
            "content": system_prompt
        }

    ]

    for message in recent_messages:

        messages.append(
            {
                "role": message["role"],
                "content": message["content"]
            }
        )

    # -----------------------------------------------------
    # Generate Response
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        try:

            with st.spinner("Thinking..."):

                completion = (
                    client.chat.completions.create(

                        model=MODEL_NAME,

                        messages=messages,

                        temperature=0.15,

                        max_tokens=5000
                    )
                )

            answer = (
                completion
                .choices[0]
                .message
                .content
            )

            if not answer:

                answer = UI_TEXT[
                    language
                ]["not_found"]

            # -------------------------------------------------
            # Display answer
            # -------------------------------------------------

            st.markdown(
                answer
            )

            # -------------------------------------------------
            # Show web search indicator
            # -------------------------------------------------

            if use_web_search and web_results:

                st.caption(
                    "🔎 Answer supported by web search results."
                )

                with st.expander("🌐 Sources"):

                    for result in web_results:

                        title = result.get(
                            "title",
                            "Source"
                        )

                        link = result.get(
                            "link",
                            ""
                        )

                        if link:

                            st.markdown(
                                f"- [{title}]({link})"
                            )

            # -------------------------------------------------
            # Save assistant response
            # -------------------------------------------------

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "web_results": web_results
            })

        except Exception as e:
            error_message = ERROR_MESSAGES.get(
                language,
                ERROR_MESSAGES["en"]
            )

            st.error(error_message)
            st.exception(e)

            print("AI Error:", e)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message
            })
