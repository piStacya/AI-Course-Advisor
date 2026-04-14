import json
import re

from openai import OpenAI

from app_logic.config import MODEL_NAME, OPENROUTER_BASE_URL


def build_system_prompt(filter_desc, context_text):
    return {
        "role": "system",
        "content": (
            "Oled Tartu Ülikooli kursuste nõustaja.\n\n"
            "REEGLID (järgi rangelt):\n"
            "1. Kasutajale kuvatakse kursused eraldi visuaalsete kastidena – ÄRGE korrake, loetlege ega nimetage ühtegi kursust, ainekoodi ega ainepealkirja oma vastuses.\n"
            "2. Kirjuta täpselt 2–3 lauset: seosta kasutaja eesmärk leitud kursuste teemavaldkonnaga. Selgita lühidalt, mis laadi aineid kastides oodata on.\n"
            "3. Kui ühtegi kursust ei leitud, ütle see ausalt ja soovita filtreid leevendada.\n"
            "4. Vasta AINULT eesti keeles.\n"
            "5. Ära lisa loendit, tabelit ega struktuurset formaati – ainult lühike loomuliku keele tekst.\n\n"
            f"Aktiivsed filtrid: {filter_desc}\n\n"
            f"[KONTEKST – ainult sinu taustateadmiseks, ära korda seda väljundis]\n{context_text}"
        )
    }


def build_messages(system_prompt, messages):
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if "debug_info" not in m
    ]
    return [system_prompt] + history


def get_stream(client, messages):
    return client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=True
    )


# --- Benchmark-specific functions ---

def build_benchmark_system_prompt(context_text):
    return {
        "role": "system",
        "content": (
            "Oled kursuste hindamise abiline. Kasuta ainult antud kursuste konteksti. "
            "Kasuta ainult välja `unique_ID` väärtusi. "
            "Ära kasuta rea numbreid, tabeli indekseid ega muid koode. "
            "Tagasta ainult lubatud `unique_ID` väärtused, mis on kontekstis selgelt ette antud. "
            'Vasta ainult kehtiva JSON-objektina kujul {"course_ids": ["ID1", "ID2"]}. '
            'Kui ükski kursus ei sobi, vasta kujul {"course_ids": []}. '
            "Ära lisa selgitusi, markdowni ega muud teksti."
            f"\n\nKursuste kontekst:\n{context_text}"
        ),
    }


def build_benchmark_user_prompt(query):
    return {
        "role": "user",
        "content": (
            "Kasuta ainult antud kursuste konteksti ning tagasta sobivate kursuste unique_ID väärtused päringu jaoks: "
            f"{query}"
        ),
    }


def create_benchmark_completion(api_key, messages):
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=False,
    )
    content = response.choices[0].message.content if response.choices else ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text").strip()
    return ""


def parse_benchmark_ids(response_text):
    normalized_ids = []
    seen = set()

    try:
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            payload = json.loads(match.group(0)) if match else None

        if isinstance(payload, list):
            raw_values = payload
        elif isinstance(payload, dict):
            for key in ("course_ids", "unique_ids", "ids", "courses", "results"):
                if isinstance(payload.get(key), list):
                    raw_values = payload[key]
                    break
            else:
                raise ValueError("No ID list found in payload")
        else:
            raise ValueError("Unexpected payload type")
    except Exception:
        pattern = r"[A-Z0-9]+\.[A-Z0-9]+\.[A-Z0-9_]+"
        raw_values = re.findall(pattern, response_text.upper())

    for item in raw_values:
        if isinstance(item, str):
            val = item
        elif isinstance(item, dict):
            val = next((item.get(k) for k in ("unique_ID", "unique_id", "course_id", "id") if isinstance(item.get(k), str)), None)
        else:
            val = None
        if val is None:
            continue
        normalized = str(val).strip().upper().replace(" ", "")
        if normalized and normalized not in seen:
            seen.add(normalized)
            normalized_ids.append(normalized)

    return normalized_ids
