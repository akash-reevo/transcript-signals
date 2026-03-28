#!/usr/bin/env python3
"""
LLM-powered report generation from final_model.json.

Reads the final model output from build_model.py and generates a human-readable
Markdown report with executive analysis, signal breakdowns, and appendices.

Library entry point:
    generate_report(output_dir, model="claude-sonnet-4-20250514", bedrock=False, aws_region=None)

Standalone:
    python3 generate_report.py --output-dir ./output [--model claude-sonnet-4-20250514] [--bedrock] [--aws-region us-west-2]

Reads:
    {output_dir}/final_model.json

Outputs:
    {output_dir}/report.md
"""

import json
import os
import sys
import argparse
from datetime import datetime, timezone

from generate_signals import create_claude_client, CostTracker, _call_claude, SIGNAL_CATEGORIES


def _build_summary(data):
    """Build the programmatic summary section (no LLM)."""
    dc = data['data_characteristics']
    perf = data['model_performance']
    det = perf['deterministic_test']

    lines = [
        "## Summary\n",
        f"- **Date generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- **Total opportunities**: {dc['total_opportunities']} ({dc['won_count']} Won, {dc['lost_count']} Lost)",
        f"- **Train/Test split**: Train {dc['train_won']}W/{dc['train_lost']}L, Test {dc['test_won']}W/{dc['test_lost']}L",
        f"- **Candidate signals**: {data.get('candidate_signal_count', 'N/A')}",
        f"- **Qualified signals (in model)**: {data.get('qualified_signal_count', 'N/A')} "
        f"({dc['variable_signals']} variable, {dc['universal_signals']} universal)",
        f"- **Deterministic test F1**: {det['f1']}",
        f"- **Monte Carlo F1**: {perf['monte_carlo']['f1']}",
        "",
        "*Data source: transcript corpus analyzed via signal evaluation pipeline.*",
    ]
    return '\n'.join(lines)


def _build_executive_summary(client, model, data, cost_tracker):
    """Generate executive summary via LLM call."""
    dc = data['data_characteristics']
    perf = data['model_performance']
    det = perf['deterministic_test']
    mc = perf['monte_carlo']

    # Top 5 signals by absolute weight
    sorted_signals = sorted(data['signals'], key=lambda s: abs(s['weight']), reverse=True)
    top_signals = sorted_signals[:5]
    top_summary = []
    for s in top_signals:
        top_summary.append({
            'name': s['name'],
            'direction': s['direction'],
            'weight': s['weight'],
            'won_pct': s['prevalence']['won_pct'],
            'lost_pct': s['prevalence']['lost_pct'],
            'delta': s['prevalence']['delta'],
        })

    system = """You are a sales analytics expert writing an executive summary for a predictive model report.
Write clear, actionable analysis for a sales leadership audience. Be specific about what the numbers mean.
Respond with JSON: {"summary": "2-3 paragraphs of analysis", "takeaway": "single most important takeaway for the sales team"}. No other text."""

    user_msg = f"""Model Performance:
- Deterministic test: Accuracy={det['accuracy']}, Precision={det['precision']}, Recall={det['recall']}, F1={det['f1']}
- Monte Carlo (1000 trials, 5% noise): Accuracy={mc['accuracy']}, Precision={mc['precision']}, Recall={mc['recall']}, F1={mc['f1']}

Data:
- {dc['total_opportunities']} opportunities ({dc['won_count']} Won, {dc['lost_count']} Lost)
- {data.get('candidate_signal_count', 'N/A')} candidate signals evaluated, {data.get('qualified_signal_count', 'N/A')} qualified for model
- {dc['variable_signals']} variable signals (predictive), {dc['universal_signals']} universal signals (non-predictive)

Top 5 signals by weight:
{json.dumps(top_summary, indent=2)}

Write 2-3 paragraphs analyzing these results, then provide a single key takeaway for the sales team."""

    try:
        result = _call_claude(client, model, system, user_msg, cost_tracker, max_tokens=2048)
        summary_text = result.get('summary', '')
        takeaway = result.get('takeaway', '')

        lines = ["## Executive Summary\n", summary_text]
        if takeaway:
            lines.extend(["", f"**Key Takeaway**: {takeaway}"])
        return '\n'.join(lines)
    except Exception as e:
        print(f"  WARNING: Executive summary generation failed: {e}")
        return "## Executive Summary\n\n*Executive summary could not be generated.*"


def _build_signals_table(signals):
    """Build the programmatic signals table (no LLM)."""
    # Only include signals in the final model, sorted by abs(weight) descending
    sorted_signals = sorted(signals, key=lambda s: abs(s['weight']), reverse=True)

    lines = [
        "## Predictive Signals\n",
        "| # | Signal | Direction | Overall % | Won % | Lost % | Delta | Weight |",
        "|---|--------|-----------|-----------|-------|--------|-------|--------|",
    ]

    for i, sig in enumerate(sorted_signals, 1):
        prev = sig['prevalence']
        direction_icon = "+" if sig['direction'] == 'positive' else "-"
        delta_str = f"{prev['delta']:+.1f}pp"
        lines.append(
            f"| {i} | {sig['name']} | {direction_icon} | "
            f"{prev['overall_pct']:.1f}% | {prev['won_pct']:.1f}% | {prev['lost_pct']:.1f}% | "
            f"{delta_str} | {sig['weight']:.4f} |"
        )

    return '\n'.join(lines)


def _format_citations(signal_entry):
    """Format citations for a signal as markdown."""
    citations = signal_entry.get('citations', {})
    correct = citations.get('correct', [])
    failures = citations.get('failures', [])

    if not correct and not failures:
        return ""

    lines = []
    if correct:
        lines.append(f"**Supporting evidence** ({len(correct)} citation{'s' if len(correct) != 1 else ''}):\n")
        for c in correct[:5]:
            cohort_label = f"Closed {c['cohort'].title()}"
            lines.append(f"> *{c['opp_name']}* ({cohort_label}, {c['meeting_date']}):")
            lines.append(f"> \"{c['snippet']}\"\n")

    if failures:
        lines.append(f"**Counter-examples** ({len(failures)} citation{'s' if len(failures) != 1 else ''}):\n")
        for c in failures[:3]:
            cohort_label = f"Closed {c['cohort'].title()}"
            lines.append(f"> *{c['opp_name']}* ({cohort_label}, {c['meeting_date']}):")
            lines.append(f"> \"{c['snippet']}\"")
            if c.get('why_misleading'):
                lines.append(f"> *Note: {c['why_misleading']}*\n")
            else:
                lines.append("")

    return '\n'.join(lines)


def _build_signal_analysis(client, model, signals, cost_tracker):
    """Build signal analysis section: LLM narratives + programmatic citations."""
    lines = ["## Signal Analysis\n"]

    # Group signals by category
    by_category = {}
    for sig in signals:
        cat = sig.get('category', 'Other')
        by_category.setdefault(cat, []).append(sig)

    for cat_name in sorted(by_category.keys()):
        cat_signals = by_category[cat_name]
        # Sort by abs(weight) within category
        cat_signals.sort(key=lambda s: abs(s['weight']), reverse=True)

        lines.append(f"### {cat_name}\n")

        # Prepare signal data for LLM
        signal_data = []
        for sig in cat_signals:
            prev = sig['prevalence']
            sample_citations = []
            for c in sig.get('citations', {}).get('correct', [])[:2]:
                sample_citations.append(c['snippet'][:150])

            signal_data.append({
                'name': sig['name'],
                'description': sig['description'],
                'direction': sig['direction'],
                'weight': sig['weight'],
                'won_pct': prev['won_pct'],
                'lost_pct': prev['lost_pct'],
                'overall_pct': prev['overall_pct'],
                'delta': prev['delta'],
                'sample_citations': sample_citations,
            })

        system = """You are a sales methodology expert analyzing predictive signals from sales transcripts.
For each signal, write:
1. A plain-English description of what this signal captures and why it matters for deal outcomes
2. An evidence paragraph interpreting the prevalence difference between won and lost deals

Be concise but insightful. Focus on actionable implications for the sales team.
Respond with JSON: {"analyses": {"SignalName": {"description": "...", "evidence": "..."}}}. No other text."""

        user_msg = f"""Category: {cat_name}

Signals to analyze:
{json.dumps(signal_data, indent=2)}

For each signal, provide a description and evidence interpretation."""

        try:
            result = _call_claude(client, model, system, user_msg, cost_tracker, max_tokens=4096)
            analyses = result.get('analyses', {})
        except Exception as e:
            print(f"  WARNING: Signal analysis for {cat_name} failed: {e}")
            analyses = {}

        for sig in cat_signals:
            prev = sig['prevalence']
            lines.append(f"#### {sig['name']}")
            direction_label = "Positive" if sig['direction'] == 'positive' else "Negative"
            lines.append(f"*{direction_label} signal | Weight: {sig['weight']:.4f} | "
                         f"Won: {prev['won_pct']:.1f}% | Lost: {prev['lost_pct']:.1f}% | "
                         f"Delta: {prev['delta']:+.1f}pp*\n")

            analysis = analyses.get(sig['name'], {})
            if analysis.get('description'):
                lines.append(analysis['description'] + "\n")
            if analysis.get('evidence'):
                lines.append(analysis['evidence'] + "\n")

            # Programmatic citations
            citation_text = _format_citations(sig)
            if citation_text:
                lines.append(citation_text)

            lines.append("")  # blank line between signals

    return '\n'.join(lines)


def _build_appendix(data):
    """Build the appendix section (programmatic, no LLM)."""
    signals = data['signals']
    opp_lists = data.get('opportunity_lists', {})

    lines = ["## Appendix\n"]

    # A: Positive signals performance
    positive = [s for s in signals if s['direction'] == 'positive' and s['tier'] == 'variable']
    positive.sort(key=lambda s: s['weight'], reverse=True)

    if positive:
        lines.append("### A. Positive Signal Performance\n")
        lines.append("| Signal | Won % | Lost % | Delta | Weight | Test F1 |")
        lines.append("|--------|-------|--------|-------|--------|---------|")
        for sig in positive:
            prev = sig['prevalence']
            f1 = sig.get('test_metrics', {}).get('f1', 'N/A')
            lines.append(f"| {sig['name']} | {prev['won_pct']:.1f}% | {prev['lost_pct']:.1f}% | "
                         f"{prev['delta']:+.1f}pp | {sig['weight']:.4f} | {f1} |")
        lines.append("")

    # B: Negative signals performance
    negative = [s for s in signals if s['direction'] == 'negative' and s['tier'] == 'variable']
    negative.sort(key=lambda s: s['weight'])

    if negative:
        lines.append("### B. Negative Signal Performance\n")
        lines.append("| Signal | Won % | Lost % | Delta | Weight | Test F1 |")
        lines.append("|--------|-------|--------|-------|--------|---------|")
        for sig in negative:
            prev = sig['prevalence']
            f1 = sig.get('test_metrics', {}).get('f1', 'N/A')
            lines.append(f"| {sig['name']} | {prev['won_pct']:.1f}% | {prev['lost_pct']:.1f}% | "
                         f"{prev['delta']:+.1f}pp | {sig['weight']:.4f} | {f1} |")
        lines.append("")

    # C: Opportunity lists
    lines.append("### C. Opportunity Lists\n")

    won_list = opp_lists.get('won', [])
    if won_list:
        lines.append(f"**Won ({len(won_list)}):**\n")
        for name in won_list:
            lines.append(f"- {name}")
        lines.append("")

    lost_list = opp_lists.get('lost', [])
    if lost_list:
        lines.append(f"**Lost ({len(lost_list)}):**\n")
        for name in lost_list:
            lines.append(f"- {name}")
        lines.append("")

    return '\n'.join(lines)


def generate_report(output_dir, model="claude-sonnet-4-20250514", bedrock=False, aws_region=None):
    """Generate a Markdown report from final_model.json.

    Args:
        output_dir: Directory containing final_model.json.
        model: Claude model for LLM-generated sections.
        bedrock: If True, use Amazon Bedrock instead of direct Anthropic API.
        aws_region: AWS region for Bedrock (default: us-west-2).

    Returns:
        dict summary with report path and cost info.
    """
    model_path = os.path.join(output_dir, 'final_model.json')
    if not os.path.isfile(model_path):
        print(f"ERROR: Final model not found at {model_path}")
        print("Run with --build-model first to generate the model.")
        sys.exit(1)

    print(f"Loading final model from {model_path}...")
    with open(model_path) as f:
        data = json.load(f)

    signal_count = len(data.get('signals', []))
    dc = data.get('data_characteristics', {})
    print(f"  {signal_count} signals, {dc.get('total_opportunities', '?')} opportunities")

    client, model = create_claude_client(bedrock=bedrock, aws_region=aws_region, model=model)
    if bedrock:
        print(f"Using Bedrock ({aws_region or 'us-west-2'}) with model {model}")
    cost_tracker = CostTracker(model)

    # Build report sections
    sections = ["# Predictive Signal Analysis Report\n"]

    print("  Building summary...")
    sections.append(_build_summary(data))

    print("  Generating executive summary (LLM)...")
    sections.append(_build_executive_summary(client, model, data, cost_tracker))

    print("  Building signals table...")
    sections.append(_build_signals_table(data['signals']))

    print("  Generating signal analysis (LLM)...")
    sections.append(_build_signal_analysis(client, model, data['signals'], cost_tracker))

    print("  Building appendix...")
    sections.append(_build_appendix(data))

    # Assemble and write
    report = '\n\n'.join(sections) + '\n'
    report_path = os.path.join(output_dir, 'report.md')
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nReport written to {report_path}")
    print(cost_tracker.summary())

    return {
        'report_path': report_path,
        'total_calls': cost_tracker.total_calls,
        'estimated_cost': cost_tracker.estimated_cost,
    }


def main():
    parser = argparse.ArgumentParser(description='Generate Markdown report from final model')
    parser.add_argument('--output-dir', required=True, help='Directory containing final_model.json')
    parser.add_argument('--model', default='claude-sonnet-4-20250514', help='Claude model (default: claude-sonnet-4-20250514)')
    parser.add_argument('--bedrock', action='store_true', help='Use Amazon Bedrock instead of direct Anthropic API')
    parser.add_argument('--aws-region', default=None, help='AWS region for Bedrock (default: us-west-2)')
    args = parser.parse_args()

    generate_report(args.output_dir, model=args.model, bedrock=args.bedrock, aws_region=args.aws_region)


if __name__ == '__main__':
    main()
