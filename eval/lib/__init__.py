"""Shared primitives for the Base dos Dados agent eval harness.

The harness is a fan-out: a runner drives the agent and writes a transcript, then
independent scorers read that transcript. This package holds the primitives those
components share, one concern per module:

    period    parse a coverage-period value ("2026", "2026-05") into grain + year
    sql       parse executed SQL (tables hit, year literals, statement shape, FROM/JOIN
              targets, CTE names)
    metadata  read a `get_table_details` output into a TableMetadata (partitioned_by,
              period_start/end, coded columns) so assertions track each run's own metadata
    gold      load, validate and index the gold YAML, plus the action / period vocabulary
    judge     the LLM judge: verdict schema, prompt, task building, dedup, one judge call

Everything here except `judge` is pure and dependency-light (stdlib + PyYAML): no agent,
no BigQuery, no LLM. `judge` calls the judge model and runs `reference_sql` on BigQuery. Import the modules explicitly (`from eval.lib import sql, metadata`) so each
call site's provenance stays clear.
"""
