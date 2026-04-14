import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import base64
import pandas as pd
import streamlit as st
from datetime import datetime
from openai import OpenAI
from PIL import Image, ImageDraw

from app_logic.data import get_models, build_linn_display_map
from app_logic.filters import simplify_keel, apply_filters, format_filters
from app_logic.retrieval import get_top_courses, build_context_text, merge_course_data
from app_logic.llm import build_system_prompt, build_messages, get_stream
from app_logic.feedback import log_feedback

# --- UI setup ---
with open("about/images.png", "rb") as _f:
    _logo_b64 = base64.b64encode(_f.read()).decode()

# Avataarid: kasutaja = sinine ruut, assistent = valge ruut sinise äärega
_user_avatar = Image.new("RGB", (32, 32), "#004099")
_asst_avatar = Image.new("RGB", (32, 32), "white")
ImageDraw.Draw(_asst_avatar).rectangle([0, 0, 31, 31], outline="#004099", width=2)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

/* ── AVATAARID: lihtsad kastid, ikoonid peidetud ── */
[data-testid="chatAvatarIcon-user"] > *,
[data-testid="chatAvatarIcon-assistant"] > * {{
    display: none !important;
}}
[data-testid="chatAvatarIcon-user"] {{
    background: white !important;
    border: 2px solid #004099 !important;
    border-radius: 4px !important;
    width: 32px !important; height: 32px !important;
    min-width: 32px !important; flex-shrink: 0 !important;
}}
[data-testid="chatAvatarIcon-assistant"] {{
    background: #004099 !important;
    border-radius: 4px !important;
    width: 32px !important; height: 32px !important;
    min-width: 32px !important; flex-shrink: 0 !important;
}}

/* ── SIDEBAR: vähenda ülemist tühja ruumi ── */
section[data-testid="stSidebar"] > div:first-child {{
    padding-top: 0 !important;
    margin-top: -1cm !important;
}}

/* ── SIDEBAR: kõik valge tekst ── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: white !important;
}}


/* Filtri pillid sidebaris — valimata */
section[data-testid="stSidebar"] button[data-testid="stBaseButton-pills"] {{
    border: 1.5px solid rgba(255,255,255,0.65) !important;
    color: white !important;
    background: transparent !important;
}}
/* Filtri pillid sidebaris — valitud */
section[data-testid="stSidebar"] button[data-testid="stBaseButton-pillsActive"],
section[data-testid="stSidebar"] button[data-testid="stBaseButton-pillsActive"] * {{
    background: #FFFFFF !important;
    box-shadow: inset 0 0 0 9999px #FFFFFF !important;
    color: #004099 !important;
    border: 1.5px solid #FFFFFF !important;
}}

/* ── PEAMINE ALA: kaardid ── */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    border-left: 3px solid #004099 !important;
    border-top: 1px solid #C4D4EC !important;
    border-right: 1px solid #C4D4EC !important;
    border-bottom: 1px solid #C4D4EC !important;
    border-radius: 6px !important;
    background: #F4F8FF !important;
}}

/* Chat sisend */
div[data-testid="stChatInput"] textarea {{
    border: 1.5px solid #004099 !important;
    border-radius: 6px !important;
    color: white !important;
}}
div[data-testid="stChatInput"] textarea::placeholder {{
    color: rgba(255,255,255,0.5) !important;
}}

/* Expander */
details {{ border: 1px solid #C4D4EC !important; border-radius: 6px !important; }}
summary {{ font-size: 0.85rem !important; color: #004099 !important; }}

/* Disabled textarea (LLM prompt) — valge tekst sinisel taustal */
textarea:disabled, textarea[disabled] {{
    color: white !important;
    -webkit-text-fill-color: white !important;
}}

/* Selectbox — nähtav nool ja äärejoon */
[data-testid="stSelectbox"] > div > div {{
    border: 1.5px solid #004099 !important;
}}
[data-testid="stSelectbox"] svg {{
    fill: #004099 !important;
    color: #004099 !important;
    opacity: 1 !important;
}}
</style>

<div style="display:flex; align-items:center; gap:12px; padding:8px 0 12px 0;
            border-bottom:2px solid #004099; margin-bottom:16px;">
  <img src="data:image/png;base64,{_logo_b64}"
       style="width:52px; height:52px; object-fit:contain; image-rendering:crisp-edges;" />
  <div>
    <div style="font-size:21px; font-weight:700; color:#004099;
                font-family:'Inter','Segoe UI',Arial,sans-serif; line-height:1.2;">
      AI Kursuse Nõustaja
    </div>
    <div style="font-size:10px; color:#5A7094; letter-spacing:1.8px;
                font-family:'Inter','Segoe UI',Arial,sans-serif; margin-top:2px;">
      TARTU ÜLIKOOL · KURSUSTE OTSING
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


embedder, df, embeddings_df, uuid_map = get_models()
linn_display_map = build_linn_display_map(df)

_linn_all = sorted(linn_display_map.keys())
_linn_priority = [l for l in ['Tartu', 'Tallinn'] if l in _linn_all]
linn_opts = _linn_priority + [l for l in _linn_all if l not in _linn_priority]


api_key = st.secrets.get("OPENROUTER_API_KEY", "")

# --- Sidebar filtrid ---
with st.sidebar:
    st.subheader("Filtrid")

    semester_opts = sorted(df['semester'].dropna().unique().tolist())
    selected_semester = st.pills("Semester", semester_opts, selection_mode="multi")

    selected_linn_display = st.pills("Asukoht", linn_opts, selection_mode="multi")
    selected_linn = [orig for d in selected_linn_display for orig in linn_display_map[d]]

    hindamine_opts = sorted(df['hindamisviis'].dropna().unique().tolist())
    selected_hindamine = st.pills("Hindamine", hindamine_opts, selection_mode="multi")

    all_oppeaste = set()
    for val in df['oppeaste'].dropna():
        for item in val.split(', '):
            all_oppeaste.add(item.strip())
    _oppeaste_priority = [o for o in ['bakalaureuseõpe', 'magistriõpe', 'doktoriõpe'] if o in all_oppeaste]
    oppeaste_opts = _oppeaste_priority + sorted(o for o in all_oppeaste if o not in _oppeaste_priority)
    selected_oppeaste = st.pills("Õppeaste", oppeaste_opts, selection_mode="multi")

    _keel_all = sorted(df['keel'].apply(simplify_keel).dropna().unique().tolist())
    _keel_priority = [k for k in ['eesti keel', 'inglise keel'] if k in _keel_all]
    keel_opts = _keel_priority + [k for k in _keel_all if k not in _keel_priority]
    selected_keel = st.pills("Õppekeel", keel_opts, selection_mode="multi")


# --- Chat ajalugu ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for i, message in enumerate(st.session_state.messages):
    _av = _user_avatar if message["role"] == "user" else _asst_avatar
    with st.chat_message(message["role"], avatar=_av):
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
                if ctx_df is not None and not ctx_df.empty:
                    display_cols = ['unique_ID', 'nimi_et', 'eap', 'semester', 'oppeaste', 'score']
                    cols_to_show = [c for c in display_cols if c in ctx_df.columns]
                    st.dataframe(ctx_df[cols_to_show], hide_index=True)
                else:
                    st.warning("Ühtegi kursust ei leitud (filtrid olid liiga karmid või andmestik tühi).")
                st.text_area("LLM-ile saadetud täpne prompt:", debug.get('system_prompt', ''), height=150, disabled=True, key=f"prompt_area_{i}")

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
                        ctx_ids = ctx_df['unique_ID'].tolist() if ctx_df is not None and not ctx_df.empty else []
                        ctx_names = ctx_df['nimi_et'].tolist() if ctx_df is not None and not ctx_df.empty and 'nimi_et' in ctx_df.columns else []
                        log_feedback(ts, debug.get('user_prompt', ''), debug.get('filters', ''), ctx_ids, ctx_names, message["content"], rating, kato)
                        st.success("Tagasiside salvestatud tagasiside_log.csv faili!")
        else:
            st.markdown(message["content"])

# --- Chat input ---
if prompt := st.chat_input("Kirjelda, mida soovid õppida..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=_user_avatar):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=_asst_avatar):
        if not api_key:
            error_msg = "Palun sisesta API võti!"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            with st.spinner("Otsin sobivaid kursusi..."):
                merged_df = merge_course_data(df, embeddings_df)
                filtered_df = apply_filters(merged_df, selected_semester, selected_linn, selected_hindamine, selected_oppeaste, selected_keel)
                filtered_count = len(filtered_df)

                if filtered_df.empty:
                    st.warning("Ühtegi kursust ei vasta filtritele.")

                results_df = get_top_courses(embedder, filtered_df, prompt)
                results_df_display = results_df.drop(columns=['embedding'], errors='ignore')
                context_text = build_context_text(results_df)
                filter_desc = format_filters(selected_semester, selected_linn_display, selected_hindamine, selected_oppeaste, selected_keel)
                system_prompt = build_system_prompt(filter_desc, context_text)
                messages_to_send = build_messages(system_prompt, st.session_state.messages)

                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
                try:
                    stream = get_stream(client, messages_to_send)
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
