import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from app_logic.config import COURSE_DATA_PATH, EMBEDDINGS_PATH, RAW_DATA_PATH, EMBEDDER_NAME
from app_logic.filters import normalize_linn


@st.cache_resource
def get_models():
    embedder = SentenceTransformer(EMBEDDER_NAME)
    df = pd.read_csv(COURSE_DATA_PATH)
    embeddings_df = pd.read_pickle(EMBEDDINGS_PATH)
    raw_df = pd.read_csv(RAW_DATA_PATH, usecols=['version__parent_code', 'latest_version_uuid'])
    uuid_map = (
        raw_df.dropna(subset=['version__parent_code'])
        .drop_duplicates('version__parent_code')
        .set_index('version__parent_code')['latest_version_uuid']
        .to_dict()
    )
    return embedder, df, embeddings_df, uuid_map


def build_linn_display_map(df):
    linn_display_map = {}
    for orig in df['linn'].dropna().unique():
        display = normalize_linn(orig)
        linn_display_map.setdefault(display, []).append(orig)
    return linn_display_map
