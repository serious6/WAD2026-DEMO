# WAD2026 Demo (uv + PydanticAI + Ollama)

Stage demo project showing structured output validation and retry behavior with a local Ollama model.

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- Local Ollama endpoint with model `llama3.2` available on `http://localhost:11434/v1`

## Install

```bash
uv sync
```

## Run

```bash
uv run python stage_demo.py
```

## What the demo shows

- `bad_model_agent` is instructed to return `fit_rating = 13`
- The schema requires `fit_rating <= 10`
- With output retries set to `2`, the run ends with a typed `validation_failed` safety response

## Optional “green” path

In `stage_demo.py`, uncomment the `@field_validator("fit_rating", mode="before")` block in `TenderEvaluation` to clamp values above 10.

Then rerun:

```bash
uv run python stage_demo.py
```

Expected: `fit_rating` is repaired to `10` and the typed response succeeds.
