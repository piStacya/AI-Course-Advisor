import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# pealkiri
st.title("🎓 AI Kursuse Nõustaja - Samm 5")
st.caption("RAG süsteem koos eel-filtreerimisega.")

# embed mudel, täisandmestik ja vektorandmebaas läheb cache'i
@st.cache_resource
def get_models():
    embedder = SentenceTransformer("BAAI/bge-m3")
    df = pd.read_csv("andmed/puhtad_andmed.csv")
    embeddings_df = pd.read_pickle("andmed/puhtad_andmed_embeddings.pkl")
    return embedder, df, embeddings_df

embedder, df, embeddings_df = get_models()

# Õppekeel: lihtsustame komaga eraldatud väärtused üheks keeleks prioriteedi järgi
# reegel: eesti keel > inglise keel > mis iganes muu
def simplify_keel(val):
    if pd.isna(val):
        return None
    langs = [l.strip() for l in val.split(', ')]
    if 'eesti keel' in langs:
        return 'eesti keel'
    if 'inglise keel' in langs:
        return 'inglise keel'
    return langs[0]

# Asukoht: eemaldame "linn" ja "alevik" järelliited ning ühendame duplikaadid
# nt "Tartu linn" ja "Tartu" → mõlemad kuvatakse kui "Tartu"
def normalize_linn(val):
    return val.replace(' linn', '').replace(' alevik', '').strip()

linn_display_map = {}  # display_nimi → [orig_väärtus1, orig_väärtus2, ...]
for orig in df['linn'].dropna().unique():
    display = normalize_linn(orig)
    linn_display_map.setdefault(display, []).append(orig)
_linn_all = sorted(linn_display_map.keys())
_linn_priority = [l for l in ['Tartu', 'Tallinn'] if l in _linn_all]
_linn_rest = [l for l in _linn_all if l not in _linn_priority]
linn_opts = _linn_priority + _linn_rest

# külgriba filtritega
with st.sidebar:
    api_key = st.text_input("OpenRouter API Key", type="password")

    st.subheader("Filtrid")

    # Semester
    semester_opts = sorted(df['semester'].dropna().unique().tolist())
    selected_semester = st.pills("Semester", semester_opts, selection_mode="multi")

    # Asukoht (normaliseeritud linn)
    selected_linn_display = st.pills("Asukoht", linn_opts, selection_mode="multi")
    # Laiendame tagasi originaalväärtusteks filtreerimiseks
    selected_linn = [orig for d in selected_linn_display for orig in linn_display_map[d]]

    # Hindamine (hindamisviis)
    hindamine_opts = sorted(df['hindamisviis'].dropna().unique().tolist())
    selected_hindamine = st.pills("Hindamine", hindamine_opts, selection_mode="multi")

    # Õppeaste – väärtused on komaga eraldatud kombinatsioonid, võtame unikaalsed üksikud
    all_oppeaste = set()
    for val in df['oppeaste'].dropna():
        for item in val.split(', '):
            all_oppeaste.add(item.strip())
    _oppeaste_priority = [o for o in ['bakalaureuseõpe', 'magistriõpe', 'doktoriõpe'] if o in all_oppeaste]
    _oppeaste_rest = sorted(o for o in all_oppeaste if o not in _oppeaste_priority)
    oppeaste_opts = _oppeaste_priority + _oppeaste_rest
    selected_oppeaste = st.pills("Õppeaste", oppeaste_opts, selection_mode="multi")

    # Õppekeel – kasutame lihtsustatud väärtusi (eesti keel > inglise keel > muu)
    _keel_all = sorted(df['keel'].apply(simplify_keel).dropna().unique().tolist())
    _keel_priority = [k for k in ['eesti keel', 'inglise keel'] if k in _keel_all]
    _keel_rest = [k for k in _keel_all if k not in _keel_priority]
    keel_opts = _keel_priority + _keel_rest
    selected_keel = st.pills("Õppekeel", keel_opts, selection_mode="multi")

# 1. alustame
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. kuvame ajaloo
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. kuulame kasutaja sõnumit
if prompt := st.chat_input("Kirjelda, mida soovid õppida..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not api_key:
            error_msg = "Palun sisesta API võti!"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            with st.spinner("Otsin sobivaid kursusi..."):
                merged_df = pd.merge(df, embeddings_df, on='unique_ID')
                filtered_df = merged_df.copy()

                if selected_semester:
                    filtered_df = filtered_df[filtered_df['semester'].isin(selected_semester)]

                if selected_linn:
                    filtered_df = filtered_df[filtered_df['linn'].isin(selected_linn)]

                if selected_hindamine:
                    filtered_df = filtered_df[filtered_df['hindamisviis'].isin(selected_hindamine)]

                if selected_oppeaste:
                    mask = filtered_df['oppeaste'].apply(
                        lambda x: any(level in str(x) for level in selected_oppeaste) if pd.notna(x) else False
                    )
                    filtered_df = filtered_df[mask]

                if selected_keel:
                    filtered_df = filtered_df[
                        filtered_df['keel'].apply(simplify_keel).isin(selected_keel)
                    ]

                #kontroll (sanity check)
                if filtered_df.empty:
                    st.warning("Ühtegi kursust ei vasta filtritele.")
                    context_text = "Sobivaid kursusi ei leitud."
                else:
                    # Arvutame sarnasuse ja sorteerime tabeli
                    query_vec = embedder.encode([prompt])[0]
                    filtered_df['score'] = cosine_similarity(
                        [query_vec], np.stack(filtered_df['embedding'])
                    )[0]

                    results_N = 5
                    results_df = filtered_df.sort_values('score', ascending=False).head(results_N)
                    results_df = results_df.drop(['score', 'embedding'], axis=1)
                    context_text = results_df.to_string()

                filter_parts = []
                if selected_semester:
                    filter_parts.append(f"semester: {', '.join(selected_semester)}")
                if selected_linn_display:
                    filter_parts.append(f"asukoht: {', '.join(selected_linn_display)}")
                if selected_hindamine:
                    filter_parts.append(f"hindamine: {', '.join(selected_hindamine)}")
                if selected_oppeaste:
                    filter_parts.append(f"õppeaste: {', '.join(selected_oppeaste)}")
                if selected_keel:
                    filter_parts.append(f"õppekeel: {', '.join(selected_keel)}")
                filter_desc = "; ".join(filter_parts) if filter_parts else "pole oluline"

                # LLM vastus
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
                system_prompt = {
                    "role": "system",
                    "content": (
                        f"Oled ülikooli kursuste nõustaja. "
                        f"Kasuta järgmisi filtreeritud kursusi ({filter_desc}):\n\n{context_text}"
                    )
                }

                messages_to_send = [system_prompt] + st.session_state.messages

                try:
                    stream = client.chat.completions.create(
                        model="google/gemma-3-27b-it",
                        messages=messages_to_send,
                        stream=True
                    )
                    response = st.write_stream(stream)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Viga: {e}")
