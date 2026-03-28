# Opportunity Transcripts CLI

Standalone CLI that connects to Snowflake, fetches meeting transcript metadata for opportunities that transitioned through specified pipeline stages, generates xlsx files, optionally downloads transcript JSON from S3, and runs predictive signal analysis.

## Setup

```bash
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt
```

## Usage — Step by step

The pipeline has 3 steps that can be run independently or combined.

### Step 1: Fetch from Snowflake + download transcripts from S3

```bash
python3 main.py \
  --org-name "Reevo GTM" \
  --pipeline "Sales" \
  --first-stages "Negotiation" \
  --final-stages "Closed Won,Closed Lost" \
  --output-dir ./output
```

Requires Snowflake env vars (`SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, etc.) and AWS credentials for S3.

### Step 2: Extract signals (LLM-based)

```bash
python3 main.py --skip-fetch \
  --final-stages "Closed Won,Closed Lost" \
  --output-dir ./output \
  --generate-signals
```

Requires `ANTHROPIC_API_KEY`. Outputs `raw_signals.json`, `signals.json`, and `signal_evaluations.json`.

### Step 3: Build model + report

```bash
python3 main.py --skip-fetch \
  --final-stages "Closed Won,Closed Lost" \
  --output-dir ./output \
  --build-model \
  --signals-file ./output/signals.json \
  --evaluations-file ./output/signal_evaluations.json
```

### Combined: Steps 2 + 3 in one run

```bash
python3 main.py --skip-fetch \
  --final-stages "Closed Won,Closed Lost" \
  --output-dir ./output \
  --generate-signals --build-model
```

Each step uses cached outputs from the previous one, so they can be run independently.

## Architecture

- `main.py` — Entry point: argparse, 4-phase orchestration (fetch → analyze → generate signals → build model), interactive disambiguation
- `snowflake_queries.py` — All Snowflake connection + query logic (6 functions)
- `gen_xlsx.py` — XLSX generation, one file per final stage (also standalone via batch JSON)
- `download_transcripts.py` — S3 download using xlsx as input (also standalone)
- `analyze_corpus.py` — Consolidate downloaded transcripts into corpus + meeting analysis (also standalone)
- `generate_signals.py` — LLM-based signal discovery, consolidation, and evaluation using Claude API (also standalone)
- `build_model.py` — Apply signals (regex or LLM evaluations), build scoring model, Monte Carlo validation (also standalone)

## Predictive signal analysis workflow

### Manual workflow (regex-based)

This is a 3-step human-in-the-loop process:

1. **Consolidate corpus** (`--analyze`): Reads transcript JSONs, filters bots, groups by opportunity, deduplicates shared recordings by content hash. Outputs `transcript_corpus.json`, `meeting_analysis.json`, `meeting_texts.json`.

2. **Human formulates signals** (manual step): Read `meeting_texts.json` and `meeting_analysis.json` to identify conversational patterns. Write a `signals.json` file with regex-based signal definitions.

3. **Build & validate model** (`--build-model --signals-file signals.json`): Applies signals, computes prevalence, 80/20 train/test split, Monte Carlo validation. Outputs `final_model.json`, `citations.json`.

Steps 1 and 3 are separate CLI invocations because step 2 happens in between (may take hours/days).

### LLM workflow (`--generate-signals`)

Replaces the manual step 2 with Claude API calls. Three phases run automatically:

1. **Signal Discovery** (Phase A): Sends each transcript to Claude to identify 30-50 candidate signals across 8 categories. Outputs `raw_signals.json`.
2. **Signal Consolidation** (Phase B): Merges/deduplicates raw signals into 80-150 universal signals. Outputs `signals.json`.
3. **Signal Evaluation** (Phase C): Evaluates each consolidated signal against each transcript semantically. Outputs `signal_evaluations.json`.

```bash
# Full LLM pipeline: generate signals + build model
python3 main.py --skip-fetch \
  --final-stages "Closed Won,Closed Lost" \
  --output-dir ./output \
  --generate-signals --build-model

# Re-build model from cached evaluations (no API calls)
python3 main.py --skip-fetch \
  --final-stages "Closed Won,Closed Lost" \
  --output-dir ./output --build-model \
  --signals-file ./output/signals.json \
  --evaluations-file ./output/signal_evaluations.json

# Force regenerate signals (ignore cache)
python3 main.py --skip-fetch \
  --final-stages "Closed Won,Closed Lost" \
  --output-dir ./output \
  --generate-signals --build-model --force-regenerate
```

**Cache files**: `raw_signals.json`, `signals.json`, `signal_evaluations.json` are cached between runs. Each phase checks input hashes (corpus SHA256 for Phase A, raw signals hash for Phase B, signals hash for Phase C) and skips if unchanged. Use `--force-regenerate` to bypass.

**Partial failure recovery**: Phases A and C track per-opportunity progress in `_raw_signals_partial.json` / `_eval_partial.json` and resume from where they left off.

**Requires**: `ANTHROPIC_API_KEY` environment variable. Uses `claude-sonnet-4-20250514` by default (override with `--claude-model`).

## Signals JSON format

### Regex format (manual workflow)

```json
[
  {
    "name": "Follow-up Call Scheduling",
    "description": "Prospect discusses scheduling a follow-up call",
    "patterns": ["follow[\\s-]?up (?:call|meeting|demo)"],
    "direction": "positive",
    "source_meeting": "08b8f81c3bef"
  }
]
```

All 5 keys required. `direction` must be "positive" or "negative". `patterns` must be valid regexes.

### LLM format (`--generate-signals`)

```json
[
  {
    "name": "Champion Identifies Internal Resistance",
    "description": "The prospect's internal champion openly discusses organizational resistance or blockers they need to overcome.",
    "category": "MEDDPICC",
    "direction": "positive",
    "source_count": 12
  }
]
```

All 4 keys (`name`, `description`, `category`, `direction`) required. No `patterns` or `source_meeting` fields. Format is auto-detected by `build_model.py`.

## Key conventions

- All Snowflake queries live in `snowflake_queries.py`, not spread across modules
- Stage names are dynamic (from `--first-stages`/`--final-stages`), never hardcoded
- Dataset counts must always be computed dynamically, never hardcoded
- Snowflake connection params come from CLI flags or env vars (`SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_ROLE`, `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_DATABASE`)
- Default authenticator is `externalbrowser` (SSO); use `--snowflake-authenticator snowflake` with `SNOWFLAKE_PASSWORD` env var for automated use
- S3 downloads use standard AWS credential chain (env vars, `~/.aws/credentials`, or IAM role)
- S3 bucket is `reevo-prod-ng-meeting-transcripts-bucket`
- `gen_xlsx.py`, `download_transcripts.py`, `analyze_corpus.py`, `generate_signals.py`, and `build_model.py` each preserve a `if __name__ == "__main__"` standalone mode
- `--skip-fetch` bypasses all Snowflake/xlsx/S3 logic; only `--final-stages` and `--output-dir` are needed
- `--build-model` implies `--analyze` if `transcript_corpus.json` doesn't exist yet
- `--generate-signals` + `--build-model` auto-wires `signals.json` and `signal_evaluations.json` from the output directory
