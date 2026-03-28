#!/usr/bin/env python3
"""
LLM-based semantic signal generation using Claude API.

Three phases:
  A) Signal Discovery — analyze each transcript for candidate signals
  B) Signal Consolidation — merge/deduplicate into 80-150 universal signals
  C) Signal Evaluation — evaluate each signal against each transcript

Library entry point:
    generate_signals(output_dir, model="claude-sonnet-4-20250514", force=False)

Standalone:
    python3 generate_signals.py --output-dir ./output [--model claude-sonnet-4-20250514] [--force]

Reads:
    {output_dir}/transcript_corpus.json

Outputs (in output_dir):
    raw_signals.json          — all candidate signals from Phase A
    signals.json              — consolidated universal signal catalog
    signal_evaluations.json   — per-opp, per-signal fires/citation data
"""

import json
import os
import sys
import time
import hashlib
import argparse
from datetime import datetime, timezone


# Cost per million tokens (approximate, for tracking)
COST_PER_M = {
    'claude-sonnet-4-20250514': {'input': 3.0, 'output': 15.0},
    'claude-sonnet-4-6': {'input': 3.0, 'output': 15.0},
    'us.anthropic.claude-sonnet-4-20250514-v1:0': {'input': 3.0, 'output': 15.0},
}
DEFAULT_COST = {'input': 3.0, 'output': 15.0}

SIGNAL_CATEGORIES = [
    "MEDDPICC (Metrics, Economic Buyer, Decision Criteria, Decision Process, Paper Process, Identify Pain, Champion, Competition)",
    "Buying Intent (urgency, timeline, budget discussion, team involvement, resource allocation)",
    "Objection Patterns (pricing pushback, competitor comparison, status quo bias, risk aversion, internal resistance)",
    "Engagement Quality (question depth, active listening, follow-up scheduling, multi-stakeholder participation)",
    "Emotional/Relationship (rapport building, enthusiasm, trust signals, skepticism, frustration)",
    "Process/Next Steps (concrete next steps, internal alignment, procurement discussion, legal/security review)",
    "Red Flags (ghosting risk, vague commitments, excessive price focus, lack of authority, stalling)",
    "Competitive Dynamics (competitor mentions, differentiation discussion, switching costs, incumbent advantage)",
]


def _sha256_of_file(path):
    """SHA256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def _sha256_of_json(obj):
    """SHA256 hex digest of a JSON-serializable object."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


class CostTracker:
    """Track cumulative API costs."""

    def __init__(self, model):
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

    def add(self, usage):
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_calls += 1

    @property
    def estimated_cost(self):
        rates = COST_PER_M.get(self.model, DEFAULT_COST)
        return (
            self.total_input_tokens / 1_000_000 * rates['input']
            + self.total_output_tokens / 1_000_000 * rates['output']
        )

    def summary(self):
        return (f"  API calls: {self.total_calls}, "
                f"tokens: {self.total_input_tokens:,} in / {self.total_output_tokens:,} out, "
                f"est. cost: ${self.estimated_cost:.2f}")


def create_claude_client(bedrock=False, aws_region=None, model="claude-sonnet-4-20250514"):
    """Create a Claude API client and resolve the model name.

    Args:
        bedrock: If True, use Amazon Bedrock instead of direct Anthropic API.
        aws_region: AWS region for Bedrock (default: us-west-2).
        model: Claude model name. Remapped for Bedrock if needed.

    Returns:
        (client, resolved_model) tuple.
    """
    if bedrock:
        from anthropic.lib.bedrock import AnthropicBedrock
        if model == "claude-sonnet-4-20250514":
            model = "us.anthropic.claude-sonnet-4-20250514-v1:0"
        client = AnthropicBedrock(aws_region=aws_region or "us-west-2")
    else:
        import anthropic
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY environment variable is required")
            sys.exit(1)
        client = anthropic.Anthropic(api_key=api_key)
    return client, model


def _call_claude(client, model, system, user_msg, cost_tracker, max_retries=5, max_tokens=8192):
    """Call Claude API with retry logic and JSON extraction.

    Returns parsed JSON dict/list from Claude's response.
    """
    backoff_times = [5, 15, 30, 60, 120]

    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            cost_tracker.add(response.usage)

            text = response.content[0].text

            # Strip markdown fences if present
            if text.strip().startswith("```"):
                lines = text.strip().split('\n')
                # Remove first line (```json or ```) and last line (```)
                if lines[-1].strip() == "```":
                    lines = lines[1:-1]
                else:
                    lines = lines[1:]
                text = '\n'.join(lines)

            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Try to find JSON array or object in the text
                for start_char, end_char in [('[', ']'), ('{', '}')]:
                    start = text.find(start_char)
                    end = text.rfind(end_char)
                    if start != -1 and end != -1 and end > start:
                        try:
                            return json.loads(text[start:end + 1])
                        except json.JSONDecodeError:
                            continue
                if attempt < max_retries:
                    print(f"    JSON parse error, retrying ({attempt + 1}/{max_retries})...")
                    continue
                raise ValueError(f"Could not parse JSON from response: {text[:200]}...")

        except Exception as e:
            err_str = str(e)
            is_overloaded = 'overloaded' in err_str.lower() or '529' in err_str
            is_rate_limit = 'rate' in err_str.lower() and not is_overloaded
            is_retryable = is_rate_limit or is_overloaded
            is_json_error = isinstance(e, ValueError) and 'JSON' in err_str

            if (is_retryable or is_json_error) and attempt < max_retries:
                wait = backoff_times[min(attempt, len(backoff_times) - 1)]
                reason = 'Rate limited' if is_rate_limit else 'API overloaded (529)' if is_overloaded else 'Parse error'
                print(f"    {reason}, retrying in {wait}s ({attempt + 1}/{max_retries})...")
                time.sleep(wait)
                continue
            raise


def _discovery_prompt(opp_name, cohort, full_text):
    """Build the system and user prompts for Phase A (signal discovery)."""
    system = f"""You are an expert sales analyst identifying behavioral signals in sales meeting transcripts.

Analyze the transcript and identify 30-50 candidate signals across these categories:
{chr(10).join(f'- {cat}' for cat in SIGNAL_CATEGORIES)}

For each signal found, provide:
- name: A clear, descriptive name (e.g., "Champion Identifies Internal Resistance")
- description: What this signal means in 1-2 sentences
- category: Which category from the list above (use the short name like "MEDDPICC", "Buying Intent", etc.)
- direction: "positive" (associated with winning) or "negative" (associated with losing)
- evidence_quote: A direct quote from the transcript (max 200 chars) showing this signal

Be specific and look for nuanced behavioral patterns, not just keyword mentions.
Focus on signals that would generalize across different sales conversations.

Respond with a JSON array of signal objects. No other text."""

    user = f"""Opportunity: {opp_name}
Outcome: Closed {cohort.title()}

Transcript:
{full_text}"""

    return system, user


def _consolidation_prompt(raw_signals):
    """Build the system and user prompts for Phase B (signal consolidation)."""
    system = """You are an expert sales analyst consolidating a large set of candidate signals into a universal catalog.

Your task:
1. Merge semantically similar signals (even if named differently across transcripts)
2. Deduplicate exact or near-exact matches
3. Generalize company-specific signals into broader patterns
4. Remove signals that are too vague to evaluate consistently
5. Target 80-150 unique signals, erring on the side of more (better to keep borderline signals — the model building step will naturally prune non-predictive ones)

For each consolidated signal, provide:
- name: Clear, descriptive name
- description: What this signal means and how to identify it (2-3 sentences, specific enough for consistent evaluation)
- category: One of: MEDDPICC, Buying Intent, Objection Patterns, Engagement Quality, Emotional/Relationship, Process/Next Steps, Red Flags, Competitive Dynamics
- direction: "positive" or "negative"
- source_count: How many raw signals were merged into this one

Make signals outcome-agnostic in their definitions — they should be identifiable regardless of whether the deal was won or lost.

Respond with a JSON array of signal objects. No other text."""

    user = f"""Here are {len(raw_signals)} raw candidate signals discovered across sales transcripts. Consolidate them into a universal signal catalog.

Raw signals:
{json.dumps(raw_signals, indent=1)}"""

    return system, user


def _evaluation_prompt(opp_name, full_text, signals):
    """Build the system and user prompts for Phase C (signal evaluation)."""
    signal_defs = []
    for sig in signals:
        signal_defs.append({
            'name': sig['name'],
            'description': sig['description'],
            'category': sig['category'],
            'direction': sig['direction'],
        })

    system = """You are an expert sales analyst evaluating whether specific behavioral signals are present in a sales transcript.

For each signal in the provided list, determine:
- fires: true if the signal is clearly present in the transcript, false otherwise
- citation: If fires is true, provide a direct quote (max 200 chars) from the transcript as evidence. If fires is false, set to null.

Be rigorous: a signal should only fire if there is clear evidence in the transcript. Do not infer signals from absence of information.

Respond with a JSON object where keys are signal names and values are {"fires": bool, "citation": string|null}. No other text."""

    user = f"""Opportunity: {opp_name}

Signals to evaluate:
{json.dumps(signal_defs, indent=1)}

Transcript:
{full_text}"""

    return system, user


def phase_a_discovery(client, model, corpus, output_dir, cost_tracker, force=False):
    """Phase A: Discover raw signals from each transcript.

    Returns list of raw signal dicts.
    """
    raw_path = os.path.join(output_dir, 'raw_signals.json')
    corpus_hash = _sha256_of_json(corpus)

    # Check cache
    if not force and os.path.isfile(raw_path):
        with open(raw_path) as f:
            cached = json.load(f)
        if cached.get('metadata', {}).get('corpus_hash') == corpus_hash:
            print(f"Phase A: Using cached raw_signals.json ({len(cached['signals'])} signals)")
            return cached['signals']
        print("Phase A: Corpus changed, regenerating...")

    print("\n=== Phase A: Signal Discovery ===")

    all_opps = {}
    for name, data in corpus.get('won', {}).items():
        all_opps[name] = {**data, 'cohort': 'won'}
    for name, data in corpus.get('lost', {}).items():
        all_opps[name] = {**data, 'cohort': 'lost'}

    # Filter out tiny transcripts
    eligible = {name: data for name, data in all_opps.items()
                if len(data.get('full_text', '')) >= 500}

    print(f"Processing {len(eligible)} opportunities (skipping {len(all_opps) - len(eligible)} with < 500 chars)")

    # Check for partial progress
    partial_signals = []
    processed_opps = set()
    partial_path = os.path.join(output_dir, '_raw_signals_partial.json')
    if not force and os.path.isfile(partial_path):
        with open(partial_path) as f:
            partial = json.load(f)
        if partial.get('corpus_hash') == corpus_hash:
            partial_signals = partial.get('signals', [])
            processed_opps = set(partial.get('processed_opps', []))
            print(f"  Resuming from partial progress: {len(processed_opps)} already processed")

    sorted_opps = sorted(eligible.keys())
    for i, opp_name in enumerate(sorted_opps, 1):
        if opp_name in processed_opps:
            continue

        opp_data = eligible[opp_name]
        cohort = opp_data['cohort']
        text_len = len(opp_data['full_text'])
        print(f"  Processing {i}/{len(sorted_opps)}: {opp_name} ({cohort}, {text_len:,} chars)...", end='', flush=True)

        system, user_msg = _discovery_prompt(opp_name, cohort, opp_data['full_text'])
        try:
            signals = _call_claude(client, model, system, user_msg, cost_tracker)
            if not isinstance(signals, list):
                print(f" ERROR: expected list, got {type(signals).__name__}")
                continue

            # Tag each signal with source info
            for sig in signals:
                sig['source_opp'] = opp_name
                sig['source_cohort'] = cohort
            partial_signals.extend(signals)
            processed_opps.add(opp_name)
            print(f" {len(signals)} signals")

        except Exception as e:
            print(f" ERROR: {e}")

        # Save partial progress every 5 opps
        if len(processed_opps) % 5 == 0:
            with open(partial_path, 'w') as f:
                json.dump({
                    'corpus_hash': corpus_hash,
                    'processed_opps': sorted(processed_opps),
                    'signals': partial_signals,
                }, f)

    # Write final output
    output = {
        'metadata': {
            'corpus_hash': corpus_hash,
            'model': model,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'opportunities_processed': len(processed_opps),
            'total_signals': len(partial_signals),
        },
        'signals': partial_signals,
    }
    with open(raw_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Clean up partial file
    if os.path.isfile(partial_path):
        os.remove(partial_path)

    print(f"\nPhase A complete: {len(partial_signals)} raw signals from {len(processed_opps)} opportunities")
    print(cost_tracker.summary())
    return partial_signals


def phase_b_consolidation(client, model, raw_signals, output_dir, cost_tracker, force=False):
    """Phase B: Consolidate raw signals into a universal catalog.

    Returns list of consolidated signal dicts.
    """
    signals_path = os.path.join(output_dir, 'signals.json')
    raw_hash = _sha256_of_json(raw_signals)

    # Check cache
    if not force and os.path.isfile(signals_path):
        with open(signals_path) as f:
            cached = json.load(f)
        # signals.json is a plain array — check companion metadata
        meta_path = os.path.join(output_dir, '_signals_meta.json')
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get('raw_hash') == raw_hash:
                print(f"Phase B: Using cached signals.json ({len(cached)} signals)")
                return cached

    print("\n=== Phase B: Signal Consolidation ===")

    # Estimate token count (~4 chars per token)
    raw_json = json.dumps(raw_signals, indent=1)
    est_tokens = len(raw_json) / 4

    if est_tokens > 150_000:
        # Split by category and consolidate in batches
        print(f"  Raw signals too large ({est_tokens:.0f} est. tokens), splitting by category...")
        by_category = {}
        for sig in raw_signals:
            cat = sig.get('category', 'Other')
            by_category.setdefault(cat, []).append(sig)

        batch_consolidated = []
        for cat, cat_signals in sorted(by_category.items()):
            print(f"  Consolidating {cat}: {len(cat_signals)} signals...", end='', flush=True)
            system, user_msg = _consolidation_prompt(cat_signals)
            result = _call_claude(client, model, system, user_msg, cost_tracker)
            if isinstance(result, list):
                batch_consolidated.extend(result)
                print(f" -> {len(result)}")
            else:
                print(f" ERROR: expected list")

        # Final merge pass
        print(f"  Final merge of {len(batch_consolidated)} signals...", end='', flush=True)
        system, user_msg = _consolidation_prompt(batch_consolidated)
        consolidated = _call_claude(client, model, system, user_msg, cost_tracker)
    else:
        print(f"  Consolidating {len(raw_signals)} raw signals...", end='', flush=True)
        system, user_msg = _consolidation_prompt(raw_signals)
        consolidated = _call_claude(client, model, system, user_msg, cost_tracker)

    if not isinstance(consolidated, list):
        print(f" ERROR: expected list, got {type(consolidated).__name__}")
        sys.exit(1)

    print(f" {len(consolidated)} consolidated signals")

    # Write signals.json (plain array for compatibility)
    with open(signals_path, 'w') as f:
        json.dump(consolidated, f, indent=2)

    # Write metadata companion
    meta_path = os.path.join(output_dir, '_signals_meta.json')
    with open(meta_path, 'w') as f:
        json.dump({
            'raw_hash': raw_hash,
            'model': model,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'signal_count': len(consolidated),
        }, f, indent=2)

    print(f"\nPhase B complete: {len(consolidated)} signals")
    print(cost_tracker.summary())
    return consolidated


def phase_c_evaluation(client, model, corpus, signals, output_dir, cost_tracker, force=False):
    """Phase C: Evaluate each signal against each transcript.

    Returns evaluations dict.
    """
    eval_path = os.path.join(output_dir, 'signal_evaluations.json')
    signals_hash = _sha256_of_json(signals)

    # Check cache
    if not force and os.path.isfile(eval_path):
        with open(eval_path) as f:
            cached = json.load(f)
        if cached.get('metadata', {}).get('signals_hash') == signals_hash:
            print(f"Phase C: Using cached signal_evaluations.json "
                  f"({len(cached.get('evaluations', {}))} opportunities)")
            return cached

    print("\n=== Phase C: Signal Evaluation ===")

    all_opps = {}
    for name, data in corpus.get('won', {}).items():
        all_opps[name] = {**data, 'cohort': 'won'}
    for name, data in corpus.get('lost', {}).items():
        all_opps[name] = {**data, 'cohort': 'lost'}

    eligible = {name: data for name, data in all_opps.items()
                if len(data.get('full_text', '')) >= 500}

    print(f"Evaluating {len(signals)} signals across {len(eligible)} opportunities")

    # Check for partial progress
    evaluations = {}
    partial_path = os.path.join(output_dir, '_eval_partial.json')
    if not force and os.path.isfile(partial_path):
        with open(partial_path) as f:
            partial = json.load(f)
        if partial.get('signals_hash') == signals_hash:
            evaluations = partial.get('evaluations', {})
            print(f"  Resuming from partial progress: {len(evaluations)} already evaluated")

    signal_names = [s['name'] for s in signals]
    sorted_opps = sorted(eligible.keys())

    for i, opp_name in enumerate(sorted_opps, 1):
        if opp_name in evaluations:
            continue

        opp_data = eligible[opp_name]
        text_len = len(opp_data['full_text'])
        print(f"  Evaluating {i}/{len(sorted_opps)}: {opp_name} ({text_len:,} chars)...", end='', flush=True)

        system, user_msg = _evaluation_prompt(opp_name, opp_data['full_text'], signals)
        try:
            result = _call_claude(client, model, system, user_msg, cost_tracker)
            if not isinstance(result, dict):
                print(f" ERROR: expected dict, got {type(result).__name__}")
                continue

            # Ensure all signals have entries (missing = fires: false)
            opp_eval = {}
            for sig_name in signal_names:
                entry = result.get(sig_name, {})
                opp_eval[sig_name] = {
                    'fires': bool(entry.get('fires', False)),
                    'citation': entry.get('citation'),
                }
            evaluations[opp_name] = opp_eval

            fires_count = sum(1 for v in opp_eval.values() if v['fires'])
            print(f" {fires_count}/{len(signals)} signals fired")

        except Exception as e:
            print(f" ERROR: {e}")

        # Save partial progress every 5 opps
        if len(evaluations) % 5 == 0:
            with open(partial_path, 'w') as f:
                json.dump({
                    'signals_hash': signals_hash,
                    'evaluations': evaluations,
                }, f)

    # Write final output
    output = {
        'metadata': {
            'signals_hash': signals_hash,
            'model': model,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'opportunities_evaluated': len(evaluations),
            'signals_evaluated': len(signals),
        },
        'evaluations': evaluations,
    }
    with open(eval_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Clean up partial file
    if os.path.isfile(partial_path):
        os.remove(partial_path)

    print(f"\nPhase C complete: {len(evaluations)} opportunities evaluated")
    print(cost_tracker.summary())
    return output


def generate_signals(output_dir, model="claude-sonnet-4-20250514", force=False,
                     bedrock=False, aws_region=None):
    """Library entry point: run all three signal generation phases.

    Args:
        output_dir: Directory containing transcript_corpus.json.
        model: Claude model to use.
        force: If True, bypass all caches.
        bedrock: If True, use Amazon Bedrock instead of direct Anthropic API.
        aws_region: AWS region for Bedrock (default: us-west-2).

    Returns:
        dict summary with counts from each phase.
    """
    client, model = create_claude_client(bedrock=bedrock, aws_region=aws_region, model=model)
    if bedrock:
        print(f"Using Bedrock ({aws_region or 'us-west-2'}) with model {model}")
    cost_tracker = CostTracker(model)

    # Load corpus
    corpus_path = os.path.join(output_dir, 'transcript_corpus.json')
    if not os.path.isfile(corpus_path):
        print(f"ERROR: Corpus not found at {corpus_path}")
        print("Run with --analyze first to generate the corpus.")
        sys.exit(1)

    print(f"Loading corpus from {corpus_path}...")
    with open(corpus_path) as f:
        corpus = json.load(f)

    won_count = len(corpus.get('won', {}))
    lost_count = len(corpus.get('lost', {}))
    print(f"Loaded {won_count} Won, {lost_count} Lost opportunities")

    # Phase A: Discovery
    raw_signals = phase_a_discovery(client, model, corpus, output_dir, cost_tracker, force)

    # Phase B: Consolidation
    consolidated = phase_b_consolidation(client, model, raw_signals, output_dir, cost_tracker, force)

    # Phase C: Evaluation
    eval_data = phase_c_evaluation(client, model, corpus, consolidated, output_dir, cost_tracker, force)

    print(f"\n=== Signal Generation Complete ===")
    print(cost_tracker.summary())

    return {
        'raw_signal_count': len(raw_signals),
        'consolidated_signal_count': len(consolidated),
        'opportunities_evaluated': len(eval_data.get('evaluations', {})),
    }


def main():
    parser = argparse.ArgumentParser(description='Generate signals from transcripts using Claude LLM')
    parser.add_argument('--output-dir', required=True, help='Directory containing transcript_corpus.json')
    parser.add_argument('--model', default='claude-sonnet-4-20250514', help='Claude model (default: claude-sonnet-4-20250514)')
    parser.add_argument('--force', action='store_true', help='Bypass all caches and regenerate')
    parser.add_argument('--bedrock', action='store_true', help='Use Amazon Bedrock instead of direct Anthropic API')
    parser.add_argument('--aws-region', default=None, help='AWS region for Bedrock (default: us-west-2)')
    args = parser.parse_args()

    generate_signals(args.output_dir, model=args.model, force=args.force,
                     bedrock=args.bedrock, aws_region=args.aws_region)


if __name__ == '__main__':
    main()
