# AI Course Advisor

A course recommendation system designed to make it easier for students to find courses that match their interests in the University of Tartu's course catalogue. Built on a RAG architecture, the app uses semantic vector search to retrieve the most relevant courses and passes them to a large language model via the OpenRouter API to generate personalised recommendations. A Streamlit interface allows users to filter courses by semester, location, study level, and language.

The assistant responds in Estonian, as the app is intended for University of Tartu students.

## Live demo

[Open app](https://ai-course-advisor.streamlit.app/)

## Code structure

| File / folder | What it does |
|---|---|
| `app.py` | App entrypoint — UI layout, sidebar filters, chat loop |
| `app_logic/config.py` | Central constants: file paths, model names, filter mappings |
| `app_logic/data.py` | Loads course data, embeddings, and the sentence transformer model |
| `app_logic/filters.py` | Applies sidebar filters to the course dataframe |
| `app_logic/retrieval.py` | Vector similarity search, top-k course retrieval |
| `app_logic/llm.py` | Builds prompts, streams LLM responses via OpenRouter |
| `app_logic/feedback.py` | Logs user feedback to CSV |
| `app_logic/benchmark.py` | Benchmark logic: runs test cases, evaluates retrieval and LLM output |
| `app_ui/benchmark.py` | Streamlit UI for the benchmark runner |
| `data/` | Course catalogue CSV and precomputed embeddings |
| `benchmark_data/` | Benchmark test cases |

## Setup

**Option A — pip:**

```bash
pip install -r requirements.txt
```

**Option B — Conda:**

```bash
conda env create -f environment.yml
conda activate oisi_projekt
```

## Running the app

```bash
streamlit run app.py
```

> On first run, the sentence transformer model will be downloaded — this may take a moment.
