import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app_logic.config import TOP_N_RESULTS, DEFAULT_EMPTY_CONTEXT


def merge_course_data(courses_df, embeddings_df):
    return pd.merge(courses_df, embeddings_df, on='unique_ID')


def get_top_courses(embedder, filtered_df, prompt, top_k=TOP_N_RESULTS):
    """Returns a ranked DataFrame (with 'score' and 'embedding' columns still present)."""
    if filtered_df.empty:
        return pd.DataFrame()

    query_vec = embedder.encode([prompt])[0]
    ranked_df = filtered_df.copy()
    ranked_df['score'] = cosine_similarity([query_vec], np.stack(ranked_df['embedding']))[0]
    return ranked_df.sort_values('score', ascending=False).head(top_k)


def build_context_text(results_df):
    """Build context string for the LLM from a results DataFrame."""
    if results_df.empty:
        return DEFAULT_EMPTY_CONTEXT

    lines = []
    for i, (_, row) in enumerate(results_df.iterrows(), start=1):
        nimi = row.get('nimi_et', '')
        nimi_en = row.get('nimi_en', '')
        uid = row.get('unique_ID', '')
        eap = row.get('eap', '')
        semester = row.get('semester', '')
        oppeaste = row.get('oppeaste', '')
        keel = row.get('keel', '')
        linn = str(row.get('linn', '')).replace(' linn', '').replace(' alevik', '')
        kirjeldus = str(row.get('kirjeldus', '')) if row.get('kirjeldus') else ''

        header = f"Kursus {i}: {uid} — {nimi}"
        if nimi_en and str(nimi_en) != 'nan':
            header += f" / {nimi_en}"

        meta_parts = []
        if eap and str(eap) != 'nan': meta_parts.append(f"{float(eap):.0f} EAP")
        if semester and str(semester) != 'nan': meta_parts.append(str(semester))
        if oppeaste and str(oppeaste) != 'nan': meta_parts.append(str(oppeaste))
        if keel and str(keel) != 'nan': meta_parts.append(str(keel))
        if linn and linn != 'nan': meta_parts.append(linn)

        lines.append(header)
        if meta_parts:
            lines.append("  " + " | ".join(meta_parts))
        if kirjeldus and kirjeldus != 'nan':
            lines.append("  Kirjeldus: " + kirjeldus[:300])
        lines.append("")

    return "\n".join(lines)
