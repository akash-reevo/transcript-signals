"""Snowflake connection and query logic for opportunity transcript extraction."""

import os
import snowflake.connector


def connect(account=None, user=None, role=None, warehouse=None, database=None, authenticator="externalbrowser"):
    """Connect to Snowflake. Parameters fall back to environment variables."""
    params = {
        "account": account or os.environ.get("SNOWFLAKE_ACCOUNT"),
        "user": user or os.environ.get("SNOWFLAKE_USER"),
        "role": role or os.environ.get("SNOWFLAKE_ROLE"),
        "warehouse": warehouse or os.environ.get("SNOWFLAKE_WAREHOUSE"),
        "database": database or os.environ.get("SNOWFLAKE_DATABASE"),
        "authenticator": authenticator,
    }
    for key in ("account", "user"):
        if not params[key]:
            raise ValueError(f"Snowflake {key} is required (pass --snowflake-{key} or set SNOWFLAKE_{key.upper()})")
    if authenticator == "snowflake":
        params["password"] = os.environ.get("SNOWFLAKE_PASSWORD", "")
        params.pop("authenticator")
    return snowflake.connector.connect(**params)


def resolve_org_id(conn, org_name):
    """Return list of {id, display_name} matching org_name (ILIKE)."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, display_name
        FROM FIVETRAN_DATABASE.POSTGRES_RDS_PUBLIC.organization
        WHERE LOWER(display_name) ILIKE %s
          AND deactivated_at IS NULL
        LIMIT 5
        """,
        (f"%{org_name}%",),
    )
    rows = cur.fetchall()
    cur.close()
    return [{"id": r[0], "display_name": r[1]} for r in rows]


def resolve_pipeline_id(conn, org_id, pipeline_name):
    """Return list of {select_list_id, display_name} matching pipeline_name (ILIKE)."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT sl.id AS select_list_id, sl.display_name AS select_list_name
        FROM FIVETRAN_DATABASE.POSTGRES_RDS_PUBLIC.select_list sl
        WHERE sl.organization_id = %s
          AND LOWER(sl.display_name) ILIKE %s
          AND sl.deleted_at IS NULL
        LIMIT 5
        """,
        (org_id, f"%{pipeline_name}%"),
    )
    rows = cur.fetchall()
    cur.close()
    return [{"select_list_id": r[0], "display_name": r[1]} for r in rows]


def find_stage_ids(conn, org_id, select_list_id, stage_names):
    """Return list of {id, display_value} for stages matching any name in stage_names."""
    like_clauses = " OR ".join(["LOWER(display_value) LIKE %s"] * len(stage_names))
    params = [org_id, select_list_id] + [f"%{name.strip().lower()}%" for name in stage_names]
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT id, display_value
        FROM FIVETRAN_DATABASE.POSTGRES_RDS_PUBLIC.select_list_value
        WHERE organization_id = %s
          AND select_list_id = %s
          AND ({like_clauses})
          AND deactivated_at IS NULL
        """,
        params,
    )
    rows = cur.fetchall()
    cur.close()
    return [{"id": r[0], "display_value": r[1]} for r in rows]


def _build_stage_like_clauses(final_stages):
    """Build LIKE clauses and params for final stage filtering."""
    clauses = " OR ".join(["LOWER(slv.display_value) LIKE %s"] * len(final_stages))
    params = [f"%{s.strip().lower()}%" for s in final_stages]
    return clauses, params


def _format_stage_ids(stage_ids):
    """Format stage IDs for IN clause. These are trusted values from prior queries."""
    return ",".join(f"'{sid}'" for sid in stage_ids)


def count_opportunities(conn, org_id, select_list_id, first_stage_ids, final_stages):
    """Return {stage_name: count} for each final stage."""
    stage_clauses, stage_params = _build_stage_like_clauses(final_stages)
    ids_csv = _format_stage_ids(first_stage_ids)

    cur = conn.cursor()
    cur.execute(
        f"""
        WITH negotiation_exits AS (
            SELECT p.id AS opportunity_id, slv.display_value AS stage_name
            FROM FIVETRAN_DATABASE.POSTGRES_RDS_PUBLIC.pipeline p
            JOIN FIVETRAN_DATABASE.POSTGRES_RDS_PUBLIC.select_list_value slv ON p.stage_id = slv.id
            JOIN FIVETRAN_DATABASE.POSTGRES_RDS_PUBLIC.pipeline_tracking pt ON p.id = pt.pipeline_id
            WHERE p.organization_id = %s
              AND p.archived_at IS NULL
              AND slv.select_list_id = %s
              AND ({stage_clauses})
              AND pt.field_name = 'stage_id'
              AND pt.from_value_uuid IN ({ids_csv})
            GROUP BY p.id, slv.display_value
        )
        SELECT stage_name, COUNT(*) AS cnt
        FROM negotiation_exits
        GROUP BY stage_name
        ORDER BY stage_name
        """,
        [org_id, select_list_id] + stage_params,
    )
    rows = cur.fetchall()
    cur.close()
    return {r[0]: r[1] for r in rows}


def fetch_transcript_data(conn, org_id, select_list_id, first_stage_ids, final_stages, num_transcripts=10, num_opportunities=None, batch_size=200):
    """Fetch all transcript rows. Returns list of dicts with keys: stage, opp, neg_date, mtg_date, s3_key."""
    stage_clauses, stage_params = _build_stage_like_clauses(final_stages)
    ids_csv = _format_stage_ids(first_stage_ids)

    # Build optional opp-limit CTE
    if num_opportunities is not None:
        opp_limit_cte = """,
            opps_ranked AS (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY stage_name ORDER BY opportunity_name) AS opp_rn
                FROM negotiation_exits
            ),
            negotiation_exits_limited AS (
                SELECT opportunity_id, opportunity_name, stage_name, negotiation_exit_date
                FROM opps_ranked WHERE opp_rn <= %s
            )"""
        opp_limit_params = [num_opportunities]
        meetings_source = "negotiation_exits_limited"
    else:
        opp_limit_cte = ""
        opp_limit_params = []
        meetings_source = "negotiation_exits"

    all_rows = []
    offset = 0
    cur = conn.cursor()

    while True:
        cur.execute(
            f"""
            WITH negotiation_exits AS (
                SELECT p.id AS opportunity_id, p.display_name AS opportunity_name,
                    slv.display_value AS stage_name,
                    MAX(DATE(pt.created_at)) AS negotiation_exit_date
                FROM FIVETRAN_DATABASE.POSTGRES_RDS_PUBLIC.pipeline p
                JOIN FIVETRAN_DATABASE.POSTGRES_RDS_PUBLIC.select_list_value slv ON p.stage_id = slv.id
                JOIN FIVETRAN_DATABASE.POSTGRES_RDS_PUBLIC.pipeline_tracking pt ON p.id = pt.pipeline_id
                WHERE p.organization_id = %s
                  AND p.archived_at IS NULL
                  AND slv.select_list_id = %s
                  AND ({stage_clauses})
                  AND pt.field_name = 'stage_id'
                  AND pt.from_value_uuid IN ({ids_csv})
                GROUP BY p.id, p.display_name, slv.display_value
            ){opp_limit_cte},
            meetings_ranked AS (
                SELECT ne.opportunity_id, ne.opportunity_name, ne.stage_name,
                    ne.negotiation_exit_date,
                    DATE(m.starts_at) AS meeting_date, t.s3_processed_key,
                    ROW_NUMBER() OVER (PARTITION BY ne.opportunity_id ORDER BY m.starts_at) AS rn
                FROM {meetings_source} ne
                JOIN FIVETRAN_DATABASE.POSTGRES_RDS_PUBLIC.meeting m
                    ON m.pipeline_id = ne.opportunity_id
                    AND DATE(m.starts_at) < ne.negotiation_exit_date
                    AND m.status = 'completed'
                    AND m.organization_id = %s
                JOIN FIVETRAN_DATABASE.POSTGRES_RDS_PUBLIC.transcript t
                    ON t.reference_id = m.latest_meeting_bot_id::TEXT
                    AND t.reference_id_type = 'meeting_bot'
            )
            SELECT stage_name, opportunity_name, negotiation_exit_date::TEXT,
                   meeting_date::TEXT, s3_processed_key
            FROM meetings_ranked WHERE rn <= %s
            ORDER BY stage_name, opportunity_name, meeting_date
            LIMIT %s OFFSET %s
            """,
            [org_id, select_list_id] + stage_params + opp_limit_params + [org_id, num_transcripts, batch_size, offset],
        )
        rows = cur.fetchall()
        for r in rows:
            all_rows.append({
                "stage": r[0],
                "opp": r[1],
                "neg_date": r[2],
                "mtg_date": r[3],
                "s3_key": r[4],
            })
        print(f"  Fetched batch at offset {offset}: {len(rows)} rows")
        if len(rows) < batch_size:
            break
        offset += batch_size

    cur.close()
    return all_rows
