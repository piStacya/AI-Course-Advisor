import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import streamlit as st
import pandas as pd
import numpy as np
import csv
import os
from datetime import datetime
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- TAGASISIDE LOGIMISE FUNKTSIOON ---
def log_feedback(timestamp, prompt, filters, context_ids, context_names, response, rating, error_category):
    file_path = 'tagasiside_log.csv'
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Aeg', 'Kasutaja päring', 'Filtrid', 'Leitud ID-d', 'Leitud ained', 'LLM Vastus', 'Hinnang', 'Veatüüp'])
        writer.writerow([timestamp, prompt, filters, str(context_ids), str(context_names), response, rating, error_category])

# pealkiri
st.title("🎓 AI Kursuse Nõustaja")
st.caption("RAG süsteem koos eel-filtreerimisega.")

# embed mudel, täisandmestik ja vektorandmebaas läheb cache'i
@st.cache_resource
def get_models():
    embedder = SentenceTransformer("BAAI/bge-m3")
    df = pd.read_csv("andmed/puhtad_andmed.csv")
    embeddings_df = pd.read_pickle("andmed/puhtad_andmed_embeddings.pkl")
    raw_df = pd.read_csv("andmed/toorandmed_aasta.csv", usecols=['version__parent_code', 'latest_version_uuid'])
    uuid_map = (
        raw_df.dropna(subset=['version__parent_code'])
        .drop_duplicates('version__parent_code')
        .set_index('version__parent_code')['latest_version_uuid']
        .to_dict()
    )
    return embedder, df, embeddings_df, uuid_map

embedder, df, embeddings_df, uuid_map = get_models()

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
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):

        if message["role"] == "assistant" and "debug_info" in message:
            debug = message["debug_info"]
            ctx_df = debug.get('context_df')

            st.markdown(message["content"])

            if ctx_df is not None and not ctx_df.empty:
                for _, row in ctx_df.iterrows():
                    with st.container(border=True):
                        col_title, col_score = st.columns([5, 1])
                        with col_title:
                            nimi_en = f" · *{row['nimi_en']}*" if pd.notna(row.get('nimi_en')) and row.get('nimi_en') else ""
                            st.markdown(f"**{row.get('nimi_et', '')}**{nimi_en}")
                        with col_score:
                            score = row.get('score')
                            if pd.notna(score):
                                st.markdown(f"<div style='text-align:right; color:gray; font-size:0.8em;'>sobivus<br><b>{float(score):.0%}</b></div>", unsafe_allow_html=True)

                        meta = []
                        if pd.notna(row.get('unique_ID')): meta.append(f"`{row['unique_ID']}`")
                        if pd.notna(row.get('eap')): meta.append(f"**{float(row['eap']):.0f} EAP**")
                        if pd.notna(row.get('semester')): meta.append(str(row['semester']))
                        if pd.notna(row.get('keel')): meta.append(str(row['keel']))
                        if pd.notna(row.get('linn')):
                            meta.append(str(row['linn']).replace(' linn', '').replace(' alevik', ''))
                        if pd.notna(row.get('veebiope')): meta.append(str(row['veebiope']))
                        if pd.notna(row.get('oppeaste')) and row.get('oppeaste'): meta.append(str(row['oppeaste']))
                        if meta:
                            st.caption(" · ".join(meta))

                        if pd.notna(row.get('kirjeldus')) and row.get('kirjeldus'):
                            st.caption(str(row['kirjeldus']))

                        uid = str(row.get('unique_ID', ''))
                        version_uuid = uuid_map.get(uid, '')
                        if uid and version_uuid:
                            st.link_button("Ava ÕIS2-s →", f"https://ois2.ut.ee/#/courses/{uid}/version/{version_uuid}/details")


            with st.expander("🔍 Vaata kapoti alla (RAG ja filtrid)"):
                st.caption(f"**Aktiivsed filtrid:** {debug.get('filters', 'Info puudub')}")
                st.write(f"Filtrid jätsid andmestikku alles **{debug.get('filtered_count', 0)}** kursust.")
                st.write("**RAG otsingu tulemus (Top 5 leitud kursust):**")
                ctx_df = debug.get('context_df')
                if ctx_df is not None and not ctx_df.empty:
                    display_cols = ['unique_ID', 'nimi_et', 'eap', 'semester', 'oppeaste', 'score']
                    cols_to_show = [c for c in display_cols if c in ctx_df.columns]
                    st.dataframe(ctx_df[cols_to_show], hide_index=True)
                else:
                    st.warning("Ühtegi kursust ei leitud (filtrid olid liiga karmid või andmestik tühi).")
                st.text_area(
                    "LLM-ile saadetud täpne prompt:",
                    debug.get('system_prompt', ''),
                    height=150,
                    disabled=True,
                    key=f"prompt_area_{i}"
                )

            with st.expander("📝 Hinda vastust (salvestab logisse)"):
                with st.form(key=f"feedback_form_{i}"):
                    rating = st.radio("Hinnang vastusele:", ["👍 Hea", "👎 Halb"], horizontal=True, key=f"rating_{i}")
                    kato = st.selectbox(
                        "Kui vastus oli halb, siis mis läks valesti?",
                        ["", "Filtrid olid liiga karmid/valed", "Otsing leidis valed ained (RAG viga)", "LLM hallutsineeris/vastas valesti"],
                        key=f"kato_{i}"
                    )
                    if st.form_submit_button("Salvesta hinnang"):
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ctx_df = debug.get('context_df')
                        ctx_ids = ctx_df['unique_ID'].tolist() if ctx_df is not None and not ctx_df.empty else []
                        ctx_names = ctx_df['nimi_et'].tolist() if ctx_df is not None and not ctx_df.empty and 'nimi_et' in ctx_df.columns else []
                        log_feedback(ts, debug.get('user_prompt', ''), debug.get('filters', ''), ctx_ids, ctx_names, message["content"], rating, kato)
                        st.success("Tagasiside salvestatud tagasiside_log.csv faili!")
        else:
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

                filtered_count = len(filtered_df)
                results_df_display = pd.DataFrame()

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
                    results_df_display = results_df.drop(columns=['embedding'], errors='ignore').copy()
                    context_text = results_df.drop(columns=['score', 'embedding'], errors='ignore').to_string()

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
                        f"Kursuste loend kuvatakse kasutajale automaatselt visuaalsete kastidena. "
                        f"KEELATUD: Ära nimeta, ära loetle ega viita ühelegi konkreetsele kursusele, aine nimele ega koodile — ei andmebaasist ega oma teadmistest. "
                        f"Kirjuta ainult 1-2 lauset üldist kommentaari: miks leitud kursused kasutaja eesmärgiga sobivad. "
                        f"Kontekst otsinguks ({filter_desc}):\n\n{context_text}"
                    )
                }

                messages_to_send = [system_prompt] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

                try:
                    stream = client.chat.completions.create(
                        model="google/gemma-3-27b-it",
                        messages=messages_to_send,
                        stream=True
                    )
                    response = st.write_stream(stream)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "debug_info": {
                            "user_prompt": prompt,
                            "filters": filter_desc,
                            "filtered_count": filtered_count,
                            "context_df": results_df_display,
                            "system_prompt": system_prompt["content"]
                        }
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Viga: {e}")
