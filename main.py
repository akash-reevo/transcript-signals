#!/usr/bin/env python3
"""
Opportunity Transcripts CLI — fetch meeting transcript S3 URLs for opportunities
that transitioned through specified stages, generate xlsx files, optionally
download transcript JSON from S3, and run predictive signal analysis.

Usage:
    # Full pipeline: fetch + download + analyze
    python3 main.py \
      --org-name "Reevo GTM" \
      --pipeline "Sales" \
      --first-stages "Negotiation" \
      --final-stages "Closed Won,Closed Lost" \
      --output-dir ./output --analyze

    # Analysis only (no Snowflake needed)
    python3 main.py --skip-fetch \
      --final-stages "Closed Won,Closed Lost" \
      --output-dir ./output --analyze

    # Build model from existing corpus + signals
    python3 main.py --skip-fetch \
      --final-stages "Closed Won,Closed Lost" \
      --output-dir ./output --build-model --signals-file ./signals.json
"""

import argparse
import os
import sys


def pick_one(matches, label, name_key):
    """Prompt user to disambiguate when multiple matches are found."""
    if len(matches) == 1:
        chosen = matches[0]
        print(f"  Resolved {label}: {chosen[name_key]}")
        return chosen
    print(f"\nMultiple {label} matches found:")
    for i, m in enumerate(matches, 1):
        print(f"  {i}. {m[name_key]}")
    while True:
        try:
            choice = int(input(f"Select {label} [1-{len(matches)}]: "))
            if 1 <= choice <= len(matches):
                return matches[choice - 1]
        except (ValueError, EOFError):
            pass
        print(f"  Please enter a number between 1 and {len(matches)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch opportunity transcript data from Snowflake, generate xlsx, download from S3, and run signal analysis."
    )
    parser.add_argument("--org-name", default=None, help="Organization display name (ILIKE match)")
    parser.add_argument("--pipeline", default=None, help="Pipeline display name (ILIKE match)")
    parser.add_argument("--first-stages", default=None, help="Comma-separated stages opps must have transitioned THROUGH")
    parser.add_argument("--final-stages", required=True, help="Comma-separated current stages to filter")
    parser.add_argument("--num-transcripts", type=int, default=10, help="Max meetings per opportunity (default: 10)")
    parser.add_argument("--num-opportunities", type=int, default=None, help="Max opportunities per final stage (default: all)")
    parser.add_argument("--output-dir", default="./output", help="Root output directory (default: ./output)")
    parser.add_argument("--batch-size", type=int, default=200, help="Rows per Snowflake batch (default: 200)")
    parser.add_argument("--skip-download", action="store_true", help="Skip S3 download step")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip Snowflake + xlsx + S3 entirely")
    parser.add_argument("--analyze", action="store_true", help="Run corpus consolidation")
    parser.add_argument("--build-model", action="store_true", help="Run model building (implies --analyze unless corpus exists)")
    parser.add_argument("--signals-file", default=None, help="Path to signals JSON (required with --build-model unless --generate-signals)")
    parser.add_argument("--evaluations-file", default=None, help="Path to pre-computed signal evaluations JSON")
    parser.add_argument("--generate-signals", action="store_true", help="Run LLM-based signal generation (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--claude-model", default="claude-sonnet-4-20250514", help="Claude model for signal generation (default: claude-sonnet-4-20250514)")
    parser.add_argument("--force-regenerate", action="store_true", help="Bypass signal generation caches")
    parser.add_argument("--exclude-prefix", default="[test", help="Filename prefix to exclude from analysis (default: [test)")
    parser.add_argument("--snowflake-account", default=None, help="Snowflake account (env: SNOWFLAKE_ACCOUNT)")
    parser.add_argument("--snowflake-user", default=None, help="Snowflake user (env: SNOWFLAKE_USER)")
    parser.add_argument("--snowflake-role", default=None, help="Snowflake role (env: SNOWFLAKE_ROLE)")
    parser.add_argument("--snowflake-warehouse", default=None, help="Snowflake warehouse (env: SNOWFLAKE_WAREHOUSE)")
    parser.add_argument("--snowflake-database", default=None, help="Snowflake database (env: SNOWFLAKE_DATABASE)")
    parser.add_argument("--snowflake-authenticator", default="externalbrowser", help="Auth method (default: externalbrowser)")
    args = parser.parse_args()

    # Validate: fetch requires org-name, pipeline, first-stages
    if not args.skip_fetch:
        missing = []
        if not args.org_name:
            missing.append("--org-name")
        if not args.pipeline:
            missing.append("--pipeline")
        if not args.first_stages:
            missing.append("--first-stages")
        if missing:
            parser.error(f"The following arguments are required when not using --skip-fetch: {', '.join(missing)}")

    # Validate: --build-model requires --signals-file unless --generate-signals will produce it
    if args.build_model and not args.signals_file and not args.generate_signals:
        # Check if a prior generate-signals run left signals.json
        auto_signals = os.path.join(args.output_dir, 'signals.json')
        if not os.path.isfile(auto_signals):
            parser.error("--build-model requires --signals-file (or use --generate-signals to create one)")

    # Validate: --generate-signals requires ANTHROPIC_API_KEY
    if args.generate_signals and not os.environ.get('ANTHROPIC_API_KEY'):
        parser.error("--generate-signals requires the ANTHROPIC_API_KEY environment variable")

    return args


def main():
    args = parse_args()

    final_stages = [s.strip() for s in args.final_stages.split(",")]

    # ── Phase 1: FETCH ──────────────────────────────────────────
    if not args.skip_fetch:
        from snowflake_queries import (
            connect,
            resolve_org_id,
            resolve_pipeline_id,
            find_stage_ids,
            count_opportunities,
            fetch_transcript_data,
        )
        from gen_xlsx import generate_xlsx_files
        from download_transcripts import download_all_transcripts

        first_stages = [s.strip() for s in args.first_stages.split(",")]

        # Step 1: Connect to Snowflake
        print("Connecting to Snowflake...")
        conn = connect(
            account=args.snowflake_account,
            user=args.snowflake_user,
            role=args.snowflake_role,
            warehouse=args.snowflake_warehouse,
            database=args.snowflake_database,
            authenticator=args.snowflake_authenticator,
        )
        print("  Connected.\n")

        try:
            # Step 2: Resolve org
            print(f"Resolving organization '{args.org_name}'...")
            orgs = resolve_org_id(conn, args.org_name)
            if not orgs:
                print(f"ERROR: No organization found matching '{args.org_name}'")
                sys.exit(1)
            org = pick_one(orgs, "organization", "display_name")
            org_id = org["id"]

            # Step 3: Resolve pipeline
            print(f"\nResolving pipeline '{args.pipeline}'...")
            pipelines = resolve_pipeline_id(conn, org_id, args.pipeline)
            if not pipelines:
                print(f"ERROR: No pipeline found matching '{args.pipeline}' in org '{org['display_name']}'")
                sys.exit(1)
            pipeline = pick_one(pipelines, "pipeline", "display_name")
            select_list_id = pipeline["select_list_id"]

            # Step 4: Find first-stage IDs
            print(f"\nFinding stage IDs for first-stages: {first_stages}...")
            first_stage_matches = find_stage_ids(conn, org_id, select_list_id, first_stages)
            if not first_stage_matches:
                print(f"ERROR: No stages found matching {first_stages} in pipeline '{pipeline['display_name']}'")
                sys.exit(1)
            first_stage_ids = [s["id"] for s in first_stage_matches]
            print("  Matched stages:")
            for s in first_stage_matches:
                print(f"    - {s['display_value']} ({s['id']})")

            # Step 5: Count opportunities per final stage
            print(f"\nCounting opportunities per final stage...")
            counts = count_opportunities(conn, org_id, select_list_id, first_stage_ids, final_stages)
            if not counts:
                print("WARNING: No opportunities found matching the criteria.")
            else:
                print("  Opportunity counts:")
                for stage, cnt in sorted(counts.items()):
                    print(f"    {stage}: {cnt}")

            # Step 6: Fetch transcript data
            print(f"\nFetching transcript data (batch_size={args.batch_size}, max {args.num_transcripts} meetings/opp)...")
            rows = fetch_transcript_data(
                conn, org_id, select_list_id,
                first_stage_ids, final_stages,
                num_transcripts=args.num_transcripts,
                num_opportunities=args.num_opportunities,
                batch_size=args.batch_size,
            )
            print(f"  Total transcript rows: {len(rows)}")

            if not rows:
                print("\nNo transcript data found. Nothing to write.")
                sys.exit(0)

            # Step 7: Generate xlsx files
            print(f"\nGenerating xlsx files in {args.output_dir}...")
            xlsx_results = generate_xlsx_files(rows, args.output_dir)

            # Step 8: Download transcripts from S3
            if not args.skip_download:
                print(f"\nDownloading transcripts from S3...")
                dl_summary = download_all_transcripts(args.output_dir, final_stages)

                print("\n=== Download Summary ===")
                for stage_name, s in dl_summary.items():
                    print(f"  {stage_name}: {s['downloaded']} downloaded, {s['skipped']} skipped, {len(s['failed'])} failed (of {s['total']} total)")
            else:
                print("\nSkipping S3 download (--skip-download).")

            # Fetch summary
            print("\n=== Fetch Summary ===")
            for stage_name, (path, count) in sorted(xlsx_results.items()):
                print(f"  {stage_name}: {count} rows -> {path}")

        finally:
            conn.close()
    else:
        print("Skipping fetch phase (--skip-fetch).\n")

    # ── Phase 2: ANALYZE ────────────────────────────────────────
    run_analyze = args.analyze
    if (args.build_model or args.generate_signals) and not args.analyze:
        # Imply --analyze if corpus doesn't exist yet
        corpus_path = os.path.join(args.output_dir, 'transcript_corpus.json')
        if not os.path.isfile(corpus_path):
            print(f"Corpus not found at {corpus_path}, running analysis first...")
            run_analyze = True

    if run_analyze:
        from analyze_corpus import consolidate_corpus

        print("\n=== Phase 2: Corpus Consolidation ===")
        corpus_summary = consolidate_corpus(args.output_dir, final_stages, args.exclude_prefix)
        print(f"\nCorpus complete: {corpus_summary['won_count']} Won, {corpus_summary['lost_count']} Lost, "
              f"{corpus_summary['unique_meetings']} unique meetings")

    # ── Phase 2.5: GENERATE SIGNALS (LLM) ───────────────────────
    if args.generate_signals:
        from generate_signals import generate_signals

        print("\n=== Phase 2.5: LLM Signal Generation ===")
        gen_summary = generate_signals(args.output_dir, model=args.claude_model, force=args.force_regenerate)
        print(f"\nSignal generation complete: {gen_summary['raw_signal_count']} raw -> "
              f"{gen_summary['consolidated_signal_count']} consolidated, "
              f"{gen_summary['opportunities_evaluated']} opportunities evaluated")

    # ── Phase 3: BUILD MODEL ────────────────────────────────────
    if args.build_model:
        from build_model import build_signal_model

        # Auto-wire signals file from generate-signals output
        signals_file = args.signals_file
        if not signals_file:
            auto_path = os.path.join(args.output_dir, 'signals.json')
            if os.path.isfile(auto_path):
                signals_file = auto_path
                print(f"Using auto-detected signals file: {signals_file}")

        # Auto-wire evaluations file from generate-signals output
        evaluations_file = args.evaluations_file
        if not evaluations_file and args.generate_signals:
            auto_eval = os.path.join(args.output_dir, 'signal_evaluations.json')
            if os.path.isfile(auto_eval):
                evaluations_file = auto_eval
                print(f"Using auto-detected evaluations file: {evaluations_file}")

        print("\n=== Phase 3: Model Building ===")
        model_summary = build_signal_model(args.output_dir, signals_file, evaluations_file=evaluations_file)
        print(f"\nModel complete: {model_summary['signal_count']} signals, "
              f"deterministic F1={model_summary['deterministic']['f1']}, "
              f"Monte Carlo F1={model_summary['monte_carlo']['f1']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
