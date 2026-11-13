"""Streamlit dashboard over the trace database: cost, tokens, latency, tool usage, errors, per-run drill-down.

    biolens dashboard            # or: streamlit run src/biolens/dashboard.py -- --home .biolens
"""

from __future__ import annotations

import argparse
import os
import sqlite3

import pandas as pd
import streamlit as st


def _home() -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=os.getenv("BIOLENS_HOME", ".biolens"))
    args, _ = ap.parse_known_args()
    return args.home


def load(db_path: str) -> tuple:
    with sqlite3.connect(db_path) as c:
        runs = pd.read_sql("SELECT * FROM runs ORDER BY created DESC", c)
        spans = pd.read_sql("SELECT * FROM spans", c)
    return runs, spans


def render(runs: pd.DataFrame, spans: pd.DataFrame):
    st.title("biolens · agent observability")
    if runs.empty or spans.empty:
        st.info("No runs yet. Try: `biolens ingest-demo` then `biolens ask \"Does zorvalin reduce relapse?\"`")
        return

    llm = spans[spans.kind == "llm"]
    tools = spans[spans.kind == "tool"]
    roots = spans[spans.parent_id.isna()]
    known = llm.cost_usd.notna().all()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Runs", len(runs))
    c2.metric("LLM calls", len(llm))
    c3.metric("Tokens", f"{int(llm.tokens_in.sum() + llm.tokens_out.sum()):,}")
    c4.metric("Cost (USD)", f"${llm.cost_usd.sum():.4f}" if known else "unknown",
              help=None if known else "Set BIOLENS_PRICING to a JSON price table to see cost.")
    c5.metric("Errors", int((spans.status == "error").sum()))

    st.subheader("Latency by span type")
    lat = spans.groupby("kind").duration_ms.describe(percentiles=[0.5, 0.95])[["count", "50%", "95%", "max"]]
    st.dataframe(lat.rename(columns={"50%": "p50 ms", "95%": "p95 ms", "max": "max ms"}).round(1))

    left, right = st.columns(2)
    with left:
        st.subheader("Tokens by model")
        by_model = llm.groupby("model")[["tokens_in", "tokens_out"]].sum()
        st.bar_chart(by_model)
    with right:
        st.subheader("Tool calls")
        if tools.empty:
            st.caption("No tool calls yet.")
        else:
            st.bar_chart(tools.groupby("name").size().rename("calls"))

    st.subheader("Runs")
    per_run = spans.groupby("run_id").agg(
        spans=("id", "count"), tokens=("tokens_in", "sum"), errors=("status", lambda s: int((s == "error").sum())))
    wall = roots.set_index("run_id").duration_ms.rename("latency_ms")
    table = runs.set_index("id")[["question", "model", "status", "stage"]].join(per_run).join(wall)
    st.dataframe(table.round(1))

    st.subheader("Run drill-down")
    pick = st.selectbox("Run", runs.id.tolist(), format_func=lambda i: f"{i} · {runs.set_index('id').question[i][:70]}")
    sel = spans[spans.run_id == pick].sort_values("start")
    depth = {}
    rows = []
    for _, s in sel.iterrows():
        depth[s.id] = depth.get(s.parent_id, -1) + 1
        rows.append({"span": "· " * depth[s.id] + f"{s.kind}:{s['name']}", "ms": round(s.duration_ms, 1),
                     "tokens_in": s.tokens_in, "tokens_out": s.tokens_out, "cost": s.cost_usd,
                     "status": s.status, "error": s.error})
    st.dataframe(pd.DataFrame(rows), width="stretch")


if __name__ == "__main__":
    home = _home()
    db = os.path.join(home, "state.db")
    if not os.path.exists(db):
        st.error(f"No state database at {db}. Run `biolens ask ...` first or pass --home.")
    else:
        render(*load(db))
