import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from app_logic.config import BENCHMARK_CASES_PATH, BENCHMARK_RUNS_PATH, DEFAULT_EMPTY_CONTEXT
from app_logic.llm import (
    build_benchmark_system_prompt,
    build_benchmark_user_prompt,
    create_benchmark_completion,
    parse_benchmark_ids,
)
from app_logic.retrieval import get_top_courses, merge_course_data


@dataclass
class BenchmarkCase:
    row_number: int
    query: str
    expected_ids: list[str]
    expects_empty: bool
    parse_error: str | None = None


@dataclass
class ComparisonResult:
    passed: bool
    missing_ids: list[str]
    unexpected_ids: list[str]


@dataclass
class StageResult:
    returned_ids: list[str]
    passed: bool
    missing_ids: list[str]
    unexpected_ids: list[str]
    raw_text: str | None = None


@dataclass
class CaseBenchmarkResult:
    case: BenchmarkCase
    retrieval: StageResult
    llm: StageResult


@dataclass
class BenchmarkRunResult:
    total_cases: int
    retrieval_correct: int
    retrieval_incorrect: int
    llm_correct: int
    llm_incorrect: int
    case_results: list[CaseBenchmarkResult]


def _normalize_course_ids(values):
    normalized_ids = []
    seen = set()

    for value in values:
        normalized_id = str(value).strip().upper()
        if normalized_id and normalized_id not in seen:
            seen.add(normalized_id)
            normalized_ids.append(normalized_id)

    return normalized_ids


def _parse_expected_ids(raw_value):
    cleaned_value = raw_value.strip()
    if cleaned_value == "-":
        return [], True

    tokens = re.split(r"[;,]", cleaned_value)
    return _normalize_course_ids(tokens), False


def _build_stage_result(case, returned_ids, raw_text=None, force_fail=False):
    comparison = compare_ids(case.expected_ids, returned_ids, case.expects_empty)
    passed = comparison.passed and not force_fail
    return StageResult(
        returned_ids=returned_ids,
        passed=passed,
        missing_ids=comparison.missing_ids,
        unexpected_ids=comparison.unexpected_ids,
        raw_text=raw_text,
    )


def _build_invalid_stage_result(case):
    raw_text = case.parse_error or "Invalid benchmark row."
    missing_ids = case.expected_ids if not case.expects_empty else []
    return StageResult(
        returned_ids=[],
        passed=False,
        missing_ids=missing_ids,
        unexpected_ids=[],
        raw_text=raw_text,
    )


def _build_context_text(ranked_df):
    if ranked_df.empty:
        return DEFAULT_EMPTY_CONTEXT

    context_columns = [
        column_name
        for column_name in ["unique_ID", "nimi_et", "eap", "semester", "oppeaste", "linn", "veebiope", "kirjeldus"]
        if column_name in ranked_df.columns
    ]
    context_df = ranked_df[context_columns].copy() if context_columns else ranked_df.drop(columns=["score", "embedding"], errors="ignore")
    if "kirjeldus" in context_df.columns:
        context_df["kirjeldus"] = context_df["kirjeldus"].fillna("").astype(str).str.slice(0, 400)

    allowed_ids = ", ".join(_normalize_course_ids(context_df["unique_ID"].tolist())) if "unique_ID" in context_df.columns else "-"
    records_json = context_df.to_json(orient="records", force_ascii=False)
    return f"Allowed unique_ID values: {allowed_ids}\n\nCourses JSON:\n{records_json}"


def _resolve_llm_ids(returned_ids, ranked_df):
    if ranked_df.empty:
        return []

    resolved_ids = []
    seen = set()
    unique_id_map = {}
    aine_kood_map = {}

    for _, row in ranked_df.iterrows():
        unique_id = str(row.get("unique_ID", "")).strip().upper()
        aine_kood = str(row.get("aine_kood", "")).strip().upper()
        base_unique_id = unique_id.split("_")[0] if unique_id else ""

        if unique_id:
            unique_id_map[unique_id] = unique_id
        if aine_kood:
            aine_kood_map.setdefault(aine_kood, [])
            if unique_id and unique_id not in aine_kood_map[aine_kood]:
                aine_kood_map[aine_kood].append(unique_id)
        if base_unique_id:
            aine_kood_map.setdefault(base_unique_id, [])
            if unique_id and unique_id not in aine_kood_map[base_unique_id]:
                aine_kood_map[base_unique_id].append(unique_id)

    for returned_id in returned_ids:
        normalized_id = str(returned_id).strip().upper()
        candidate_ids = []

        if normalized_id in unique_id_map:
            candidate_ids = [unique_id_map[normalized_id]]
        elif normalized_id in aine_kood_map:
            candidate_ids = aine_kood_map[normalized_id][:1]

        for candidate_id in candidate_ids:
            if candidate_id not in seen:
                seen.add(candidate_id)
                resolved_ids.append(candidate_id)

    return resolved_ids


def load_benchmark_cases(path: str = BENCHMARK_CASES_PATH) -> list[BenchmarkCase]:
    cases = []

    with open(path, newline="", encoding="utf-8") as file_handle:
        reader = csv.reader(file_handle)
        next(reader, None)

        for row_number, row in enumerate(reader, start=2):
            if len(row) < 2:
                query = row[0].strip() if row else ""
                cases.append(
                    BenchmarkCase(
                        row_number=row_number,
                        query=query,
                        expected_ids=[],
                        expects_empty=False,
                        parse_error="Row has fewer than 2 columns.",
                    )
                )
                continue

            expected_ids, expects_empty = _parse_expected_ids(row[1])
            parse_error = None
            if not expects_empty and not expected_ids:
                parse_error = "Expected ID column is empty or invalid."

            cases.append(
                BenchmarkCase(
                    row_number=row_number,
                    query=row[0].strip(),
                    expected_ids=expected_ids,
                    expects_empty=expects_empty,
                    parse_error=parse_error,
                )
            )

    return cases


def compare_ids(expected_ids, actual_ids, expects_empty) -> ComparisonResult:
    normalized_expected = _normalize_course_ids(expected_ids)
    normalized_actual = _normalize_course_ids(actual_ids)

    if expects_empty:
        return ComparisonResult(
            passed=len(normalized_actual) == 0,
            missing_ids=[],
            unexpected_ids=normalized_actual,
        )

    missing_ids = [course_id for course_id in normalized_expected if course_id not in normalized_actual]
    unexpected_ids = [course_id for course_id in normalized_actual if course_id not in normalized_expected]
    return ComparisonResult(
        passed=len(missing_ids) == 0,
        missing_ids=missing_ids,
        unexpected_ids=unexpected_ids,
    )


def evaluate_case_retrieval(case, embedder, merged_df) -> StageResult:
    if case.parse_error:
        return _build_invalid_stage_result(case)

    ranked_df = get_top_courses(embedder, merged_df, case.query, top_k=5)
    returned_ids = _normalize_course_ids(ranked_df["unique_ID"].tolist()) if "unique_ID" in ranked_df.columns else []
    return _build_stage_result(case, returned_ids)


def evaluate_case_llm(case, api_key, ranked_df, llm_messages_context=None) -> StageResult:
    if case.parse_error:
        return _build_invalid_stage_result(case)

    messages = llm_messages_context
    if messages is None:
        context_text = _build_context_text(ranked_df)
        messages = [
            build_benchmark_system_prompt(context_text),
            build_benchmark_user_prompt(case.query),
        ]

    try:
        response_text = create_benchmark_completion(api_key, messages)
    except Exception as error:
        return StageResult(
            returned_ids=[],
            passed=False,
            missing_ids=case.expected_ids if not case.expects_empty else [],
            unexpected_ids=[],
            raw_text=str(error),
        )

    try:
        returned_ids = parse_benchmark_ids(response_text)
        returned_ids = _resolve_llm_ids(returned_ids, ranked_df)
    except Exception:
        return StageResult(
            returned_ids=[],
            passed=False,
            missing_ids=case.expected_ids if not case.expects_empty else [],
            unexpected_ids=[],
            raw_text=response_text,
        )

    return _build_stage_result(case, returned_ids, raw_text=response_text)


def run_benchmark_suite(cases, embedder, courses_df, embeddings_df, api_key, case_limit=None, progress_callback=None) -> BenchmarkRunResult:
    selected_cases = cases if case_limit is None else cases[:case_limit]
    merged_df = merge_course_data(courses_df, embeddings_df)
    case_results = []
    total_cases = len(selected_cases)

    if progress_callback is not None:
        progress_callback(0, total_cases, None)

    for index, case in enumerate(selected_cases, start=1):
        ranked_df = get_top_courses(embedder, merged_df, case.query, top_k=5) if not case.parse_error else pd.DataFrame()
        retrieval_ids = _normalize_course_ids(ranked_df["unique_ID"].tolist()) if "unique_ID" in ranked_df.columns else []
        retrieval_result = _build_stage_result(case, retrieval_ids) if not case.parse_error else _build_invalid_stage_result(case)

        context_text = _build_context_text(ranked_df)
        benchmark_messages = [
            build_benchmark_system_prompt(context_text),
            build_benchmark_user_prompt(case.query),
        ]
        llm_result = evaluate_case_llm(case, api_key, ranked_df, benchmark_messages)

        case_results.append(
            CaseBenchmarkResult(
                case=case,
                retrieval=retrieval_result,
                llm=llm_result,
            )
        )

        if progress_callback is not None:
            progress_callback(index, total_cases, case)

    retrieval_correct = sum(result.retrieval.passed for result in case_results)
    llm_correct = sum(result.llm.passed for result in case_results)
    total_cases = len(case_results)

    return BenchmarkRunResult(
        total_cases=total_cases,
        retrieval_correct=retrieval_correct,
        retrieval_incorrect=total_cases - retrieval_correct,
        llm_correct=llm_correct,
        llm_incorrect=total_cases - llm_correct,
        case_results=case_results,
    )


def _case_from_dict(payload):
    return BenchmarkCase(**payload)


def _stage_from_dict(payload):
    return StageResult(**payload)


def _case_result_from_dict(payload):
    return CaseBenchmarkResult(
        case=_case_from_dict(payload["case"]),
        retrieval=_stage_from_dict(payload["retrieval"]),
        llm=_stage_from_dict(payload["llm"]),
    )


def serialize_benchmark_run(benchmark_results, saved_at=None):
    return {
        "saved_at": saved_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results": asdict(benchmark_results),
    }


def deserialize_benchmark_run(payload):
    results_payload = payload["results"]
    benchmark_results = BenchmarkRunResult(
        total_cases=results_payload["total_cases"],
        retrieval_correct=results_payload["retrieval_correct"],
        retrieval_incorrect=results_payload["retrieval_incorrect"],
        llm_correct=results_payload["llm_correct"],
        llm_incorrect=results_payload["llm_incorrect"],
        case_results=[_case_result_from_dict(item) for item in results_payload["case_results"]],
    )
    return benchmark_results, payload.get("saved_at")


def save_benchmark_run(benchmark_results, path=BENCHMARK_RUNS_PATH):
    file_path = Path(path)
    saved_run = serialize_benchmark_run(benchmark_results)
    runs = []

    if file_path.exists():
        with open(file_path, encoding="utf-8") as file_handle:
            runs = json.load(file_handle)
        if not isinstance(runs, list):
            runs = []

    runs.append(saved_run)
    with open(file_path, "w", encoding="utf-8") as file_handle:
        json.dump(runs, file_handle, ensure_ascii=False, indent=2)

    return saved_run["saved_at"]


def load_last_benchmark_run(path=BENCHMARK_RUNS_PATH):
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(path)

    with open(file_path, encoding="utf-8") as file_handle:
        runs = json.load(file_handle)

    if not isinstance(runs, list) or not runs:
        raise ValueError("No saved benchmark runs found.")

    return deserialize_benchmark_run(runs[-1])
