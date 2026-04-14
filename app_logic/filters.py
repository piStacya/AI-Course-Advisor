import pandas as pd


def simplify_keel(val):
    if pd.isna(val):
        return None
    langs = [l.strip() for l in val.split(', ')]
    if 'eesti keel' in langs:
        return 'eesti keel'
    if 'inglise keel' in langs:
        return 'inglise keel'
    return langs[0]


def normalize_linn(val):
    return val.replace(' linn', '').replace(' alevik', '').strip()


def apply_filters(merged_df, selected_semester, selected_linn, selected_hindamine, selected_oppeaste, selected_keel):
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

    return filtered_df


def format_filters(selected_semester, selected_linn_display, selected_hindamine, selected_oppeaste, selected_keel):
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
    return "; ".join(filter_parts) if filter_parts else "pole oluline"
