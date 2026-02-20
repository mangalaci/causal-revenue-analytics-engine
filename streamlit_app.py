"""
Causal Revenue Analytics Engine — Streamlit Dashboard
Stage 1 only: PVM (Price-Volume-Mix) decomposition from CSV data.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analyzer_from_csv_v2 import (
    AnalyzerConfig,
    build_llm_input_v2,
)

DATA_CSV = "abbb.csv.gz"


# ------------------------------------------------------------------
# Helper: deterministic action suggestion (extracted from llm_v2.py)
# ------------------------------------------------------------------
def _suggest_action_from_pv_margin(
    price_effect: float, volume_effect: float, delta_margin: float
) -> str:
    pe = price_effect or 0.0
    ve = volume_effect or 0.0
    dm = delta_margin or 0.0

    if pe > 0 and ve < 0:
        if dm < 0:
            return "PRICE DOWN TEST / PROMO"
        if dm > 0:
            return "KEEP / PRICE-UP (controlled)"
        return "MONITOR (borderline)"

    if pe < 0 and ve > 0:
        if dm > 0:
            return "PROMO SCALE"
        if dm < 0:
            return "PROMO STOP / PRICE UP TEST"
        return "MONITOR"

    if pe < 0 and ve < 0:
        if dm < 0:
            return "DELIST / FIX AVAILABILITY"
        return "PRICE REVIEW"

    if pe > 0 and ve > 0:
        if dm > 0:
            return "KEEP / PRICE-UP"
        return "KEEP (watch margin)"

    return "MONITOR"


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------
def fmt_money(x) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    s = f"{int(round(v)):,}".replace(",", " ")
    return f"{s} Ft"


def fmt_pct(x) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Causal Revenue Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("Causal Revenue Analytics Engine")
st.caption("PVM (Price-Volume-Mix) decomposition from CSV data")


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.header("Configuration")

st.sidebar.subheader("Period Settings")

today = date.today()
first_of_month = today.replace(day=1)
prev_month_end = first_of_month - timedelta(days=1)
prev_month_start = prev_month_end.replace(day=1)
two_months_ago_end = prev_month_start - timedelta(days=1)
two_months_ago_start = two_months_ago_end.replace(day=1)

t1_start = st.sidebar.date_input("t1 start", value=two_months_ago_start)
t1_end = st.sidebar.date_input("t1 end", value=two_months_ago_end)
t2_start = st.sidebar.date_input("t2 start", value=prev_month_start)
t2_end = st.sidebar.date_input("t2 end", value=prev_month_end)

origin = st.sidebar.text_input("Origin filter", value="invoices")
csv_encoding = st.sidebar.selectbox("CSV encoding", ["cp1250", "utf-8", "latin-1", "iso-8859-2"], index=0)

run_btn = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)


# ------------------------------------------------------------------
# Main logic
# ------------------------------------------------------------------
if not run_btn and "result" not in st.session_state:
    st.info("Configure periods and click **Run Analysis**.")
    st.stop()

if run_btn:
    data_dir = str(Path(__file__).parent)
    csv_path = Path(data_dir) / DATA_CSV
    if not csv_path.exists():
        st.error(f"Data file not found: {DATA_CSV}")
        st.stop()

    cfg = AnalyzerConfig(
        data_dir=data_dir,
        input_csv=DATA_CSV,
        inventory_csv=None,
        t1_start=str(t1_start),
        t1_end=str(t1_end),
        t2_start=str(t2_start),
        t2_end=str(t2_end),
        origin=origin,
        csv_encoding=csv_encoding,
        csv_engine="python",
        csv_on_bad_lines="skip",
    )

    with st.spinner("Running PVM analysis..."):
        try:
            result = build_llm_input_v2(cfg)
            st.session_state["result"] = result
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

result = st.session_state.get("result")
if result is None:
    st.stop()


# ------------------------------------------------------------------
# Extract key data from result
# ------------------------------------------------------------------
global_kpis = result.get("global_kpis", {})
bridge = result.get("bridge", {})
common_pvm = result.get("common_pvm", {})
math_check = result.get("math_check", {})
periods = result.get("periods", {})
quadrants = result.get("quadrants", {})
quad_summary = quadrants.get("summary", [])
quad_impact = result.get("quadrant_impact_contrib", {})
quad_margin = result.get("quadrant_margin_contrib", {})
quad_key = result.get("quadrant_key_products", {})
volume_explanation = result.get("volume_explanation", {})
price_explanation = result.get("price_explanation", {})
margin_kpis = result.get("margin_kpis", {})


# ------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Quadrant Analysis",
    "Key Decision Products",
    "Volume & Price Deep-dive",
    "Margin",
])


# ==================================================================
# TAB 1 — Overview
# ==================================================================
with tab1:
    st.subheader("Revenue Overview")

    t1_label = f"{periods.get('t1', {}).get('start', '')} – {periods.get('t1', {}).get('end', '')}"
    t2_label = f"{periods.get('t2', {}).get('start', '')} – {periods.get('t2', {}).get('end', '')}"

    col1, col2, col3 = st.columns(3)
    t1_rev = global_kpis.get("t1_revenue", 0)
    t2_rev = global_kpis.get("t2_revenue", 0)
    delta_rev = global_kpis.get("delta_revenue", 0)

    col1.metric(f"t1 Revenue ({t1_label})", fmt_money(t1_rev))
    col2.metric(f"t2 Revenue ({t2_label})", fmt_money(t2_rev))
    col3.metric("Delta Revenue", fmt_money(delta_rev), delta=fmt_money(delta_rev))

    # Math check badge
    if math_check.get("audit_passes"):
        st.success(f"Math check: PASS (bridge vs global diff = {fmt_money(math_check.get('bridge_vs_global_diff'))})")
    else:
        st.warning(f"Math check: FAIL (bridge vs global diff = {fmt_money(math_check.get('bridge_vs_global_diff'))}, tolerance = {fmt_money(math_check.get('tolerance'))})")

    st.divider()

    # Revenue Bridge bar chart
    col_bridge, col_pvm = st.columns(2)

    with col_bridge:
        st.subheader("Revenue Bridge")
        bridge_labels = ["Appearing (t2-only)", "Disappearing (t1-only)", "Common Delta"]
        bridge_values = [
            bridge.get("appearing_total", 0),
            -bridge.get("disappearing_total", 0),
            bridge.get("common_delta_total", 0),
        ]
        bridge_colors = ["#2ca02c", "#d62728", "#1f77b4"]

        fig_bridge = go.Figure(go.Bar(
            x=bridge_labels,
            y=bridge_values,
            marker_color=bridge_colors,
            text=[fmt_money(v) for v in bridge_values],
            textposition="outside",
        ))
        fig_bridge.update_layout(
            yaxis_title="Revenue (Ft)",
            height=400,
            margin=dict(t=20),
        )
        st.plotly_chart(fig_bridge, use_container_width=True)

    with col_pvm:
        st.subheader("PVM Waterfall (Common Products)")

        pvm_labels = ["Volume", "Price", "Mix", "Total ΔR"]
        pvm_values = [
            common_pvm.get("volume_effect", 0),
            common_pvm.get("price_effect", 0),
            common_pvm.get("mix_effect", 0),
            common_pvm.get("delta_common_revenue", 0),
        ]

        fig_pvm = go.Figure(go.Waterfall(
            x=pvm_labels,
            y=pvm_values,
            measure=["relative", "relative", "relative", "total"],
            text=[fmt_money(v) for v in pvm_values],
            textposition="outside",
            connector_line_color="grey",
            increasing_marker_color="#2ca02c",
            decreasing_marker_color="#d62728",
            totals_marker_color="#1f77b4",
        ))
        fig_pvm.update_layout(
            yaxis_title="Revenue (Ft)",
            height=400,
            margin=dict(t=20),
        )
        st.plotly_chart(fig_pvm, use_container_width=True)


# ==================================================================
# TAB 2 — Quadrant Analysis
# ==================================================================
with tab2:
    st.subheader("Price x Volume Quadrant Analysis")

    if quad_summary:
        rows = []
        for q in quad_summary:
            rows.append({
                "Quadrant": q.get("quadrant", ""),
                "Pack Count": int(q.get("pack_cnt", 0)),
                "t1 Qty": int(q.get("t1_qty", 0)),
                "t2 Qty": int(q.get("t2_qty", 0)),
                "Δ Qty": int(q.get("delta_qty", 0)),
                "t1 Rev": fmt_money(q.get("t1_rev")),
                "t2 Rev": fmt_money(q.get("t2_rev")),
                "Δ Rev": fmt_money(q.get("delta_rev")),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # Quadrant impact contribution
    st.subheader("Quadrant Impact Contribution")

    reconciliation = quad_impact.pop("_reconciliation", None)

    for quad_name, qdata in quad_impact.items():
        if quad_name.startswith("_"):
            continue
        with st.expander(f"{quad_name} — Impact: {fmt_money(qdata.get('impact_total'))}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Packs", qdata.get("pack_cnt", 0))
            c2.metric("Volume Total", fmt_money(qdata.get("volume_total")))
            c3.metric("Price Total", fmt_money(qdata.get("price_total")))
            c4.metric("ΔR Total", fmt_money(qdata.get("delta_rev_total")))

            top_products = qdata.get("top_products", [])
            if top_products:
                tp_rows = []
                for p in top_products[:10]:
                    tp_rows.append({
                        "CT2_pack": p.get("CT2_pack", ""),
                        "Impact": fmt_money(p.get("impact_effect")),
                        "Price Effect": fmt_money(p.get("price_effect")),
                        "Volume Effect": fmt_money(p.get("volume_effect")),
                        "ΔR": fmt_money(p.get("delta_rev")),
                        "Share %": fmt_pct(p.get("share_pct")),
                    })
                st.dataframe(pd.DataFrame(tp_rows), use_container_width=True, hide_index=True)

    # Reconciliation
    if reconciliation:
        st.divider()
        st.subheader("Reconciliation")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Common ΔR Total", fmt_money(reconciliation.get("common_delta_rev_total")))
        rc2.metric("Σ Volume", fmt_money(reconciliation.get("sum_quadrant_volume_total")))
        rc3.metric("Σ Price", fmt_money(reconciliation.get("sum_quadrant_price_total")))
        rc4.metric("Residual (Mix)", fmt_money(reconciliation.get("residual_mix_total")))

    # Put it back for other tabs
    if reconciliation:
        quad_impact["_reconciliation"] = reconciliation


# ==================================================================
# TAB 3 — Key Decision Products
# ==================================================================
with tab3:
    st.subheader("Key Decision Products")
    st.caption("Per-quadrant top products by impact, with suggested actions based on PV + margin signals")

    quad_order = [
        "D1: Price↑ Volume↑",
        "D2: Price↑ Volume↓",
        "D3: Price↓ Volume↑",
        "D4: Price↓ Volume↓",
        "Neutral / No change",
    ]

    for quad_name in quad_order:
        qdata = quad_key.get(quad_name)
        if not qdata:
            continue

        items = qdata.get("items", [])
        coverage = qdata.get("coverage_pct")

        if not items:
            continue

        coverage_text = f" — Coverage: {coverage:.1f}%" if coverage is not None else ""
        with st.expander(f"{quad_name}{coverage_text}", expanded=(quad_name.startswith("D4") or quad_name.startswith("D1"))):
            rows = []
            for it in items:
                pe = it.get("price_effect", 0)
                ve = it.get("volume_effect", 0)
                dm = it.get("delta_margin", 0)
                action = _suggest_action_from_pv_margin(pe, ve, dm)
                rows.append({
                    "CT2_pack": it.get("CT2_pack", ""),
                    "Price Effect": fmt_money(pe),
                    "Volume Effect": fmt_money(ve),
                    "Impact": fmt_money(it.get("impact_effect")),
                    "Margin Δ": fmt_money(dm),
                    "Suggested Action": action,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ==================================================================
# TAB 4 — Volume & Price Deep-dive
# ==================================================================
with tab4:
    st.subheader("Volume Explanation")

    # Basket KPIs
    basket = volume_explanation.get("basket", {})
    if basket:
        st.markdown("**Basket KPIs**")
        bc1, bc2, bc3 = st.columns(3)

        t1_basket = basket.get("t1", {})
        t2_basket = basket.get("t2", {})
        delta_basket = basket.get("delta", {})

        bc1.metric("Invoice Count (t1 → t2)",
                   f"{t1_basket.get('invoice_cnt', 'n/a')} → {t2_basket.get('invoice_cnt', 'n/a')}",
                   delta=str(delta_basket.get("invoice_cnt", "")))
        bc2.metric("Units/Invoice (t1 → t2)",
                   f"{t1_basket.get('units_per_invoice', 'n/a')} → {t2_basket.get('units_per_invoice', 'n/a')}",
                   delta=str(delta_basket.get("units_per_invoice", "")))
        bc3.metric("Interpretation", basket.get("interpretation", "n/a"))

    st.divider()

    # Buyer split
    buyer_split = volume_explanation.get("buyer_split", {})
    vol_contrib = volume_explanation.get("volume_pack_contrib", {})

    if buyer_split:
        st.markdown("**Buyer Split (Repeat vs One-time)**")
        for seg in ["one_time", "repeat"]:
            seg_data = buyer_split.get(seg)
            if not seg_data:
                continue

            st.markdown(f"***{seg}***")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Orders (t1 → t2)",
                       f"{seg_data.get('t1_orders', 'n/a')} → {seg_data.get('t2_orders', 'n/a')}",
                       delta=str(seg_data.get("delta_orders", "")))
            sc2.metric("Volume Effect", fmt_money(seg_data.get("volume_effect_total")))
            sc3.metric("Share of Total Volume", fmt_pct(seg_data.get("share_of_total_volume_effect_pct")))

            # Top packs per segment
            seg_vol = vol_contrib.get(seg, {})
            top_neg = seg_vol.get("top_packs_neg", [])
            top_pos = seg_vol.get("top_packs_pos", [])

            if top_neg or top_pos:
                neg_col, pos_col = st.columns(2)
                with neg_col:
                    if top_neg:
                        st.markdown("Top negative contributors:")
                        neg_rows = [
                            {"CT2_pack": p.get("CT2_pack"), "Volume Effect": fmt_money(p.get("volume_effect")), "Share": fmt_pct(p.get("share"))}
                            for p in top_neg[:3]
                        ]
                        st.dataframe(pd.DataFrame(neg_rows), use_container_width=True, hide_index=True)
                with pos_col:
                    if top_pos:
                        st.markdown("Top positive contributors:")
                        pos_rows = [
                            {"CT2_pack": p.get("CT2_pack"), "Volume Effect": fmt_money(p.get("volume_effect")), "Share": fmt_pct(p.get("share"))}
                            for p in top_pos[:3]
                        ]
                        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

    st.divider()

    # Price explanation
    st.subheader("Price Explanation")
    price_contrib = price_explanation.get("price_pack_contrib", {})

    if price_contrib:
        total_price_eff = common_pvm.get("price_effect")
        denom = abs(total_price_eff) if (total_price_eff and abs(total_price_eff) > 0) else None

        for seg in ["one_time", "repeat"]:
            seg_obj = price_contrib.get(seg, {})
            if not seg_obj:
                continue

            seg_total = seg_obj.get("segment_price_effect_total")
            share_total = None
            if denom and seg_total is not None:
                share_total = (abs(float(seg_total)) / float(denom)) * 100.0

            st.markdown(f"***{seg}*** — Price Effect: {fmt_money(seg_total)} ({fmt_pct(share_total)})")

            top_neg = seg_obj.get("top_packs_neg", [])
            top_pos = seg_obj.get("top_packs_pos", [])

            if top_neg or top_pos:
                neg_col, pos_col = st.columns(2)
                with neg_col:
                    if top_neg:
                        st.markdown("Top negative contributors:")
                        neg_rows = [
                            {"CT2_pack": p.get("CT2_pack"), "Price Effect": fmt_money(p.get("price_effect")), "Share": fmt_pct(p.get("share"))}
                            for p in top_neg[:3]
                        ]
                        st.dataframe(pd.DataFrame(neg_rows), use_container_width=True, hide_index=True)
                with pos_col:
                    if top_pos:
                        st.markdown("Top positive contributors:")
                        pos_rows = [
                            {"CT2_pack": p.get("CT2_pack"), "Price Effect": fmt_money(p.get("price_effect")), "Share": fmt_pct(p.get("share"))}
                            for p in top_pos[:3]
                        ]
                        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)


# ==================================================================
# TAB 5 — Margin
# ==================================================================
with tab5:
    st.subheader("Margin KPIs")

    if margin_kpis:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("t1 Margin", fmt_money(margin_kpis.get("t1_margin")))
        mc2.metric("t2 Margin", fmt_money(margin_kpis.get("t2_margin")))
        mc3.metric("Δ Margin", fmt_money(margin_kpis.get("delta_margin")),
                   delta=fmt_money(margin_kpis.get("delta_margin")))

        t1_pct = margin_kpis.get("t1_margin_pct")
        t2_pct = margin_kpis.get("t2_margin_pct")
        if t1_pct is not None and t2_pct is not None:
            mc4.metric("Margin %", f"{t2_pct:.2%}", delta=f"{(t2_pct - t1_pct):.2%}")

    st.divider()

    # Margin by quadrant
    st.subheader("Margin by Quadrant")

    for quad_name in quad_order:
        qm = quad_margin.get(quad_name)
        if not qm:
            continue

        with st.expander(f"{quad_name} — Margin Δ: {fmt_money(qm.get('quadrant_delta_margin_total'))}"):
            qmc1, qmc2, qmc3 = st.columns(3)
            qmc1.metric("Packs", qm.get("pack_cnt", 0))
            qmc2.metric("Total Margin Δ", fmt_money(qm.get("quadrant_delta_margin_total")))

            improver = qm.get("top_improver")
            detractor = qm.get("top_detractor")

            if improver:
                qmc3.metric("Top Improver", improver.get("CT2_pack", ""), delta=fmt_money(improver.get("delta_margin")))

            if detractor:
                st.metric("Top Detractor", detractor.get("CT2_pack", ""), delta=fmt_money(detractor.get("delta_margin")))

            top_by_abs = qm.get("top_products_by_abs", [])
            if top_by_abs:
                margin_rows = [
                    {"CT2_pack": p.get("CT2_pack", ""), "Margin Δ": fmt_money(p.get("delta_margin"))}
                    for p in top_by_abs[:10]
                ]
                st.dataframe(pd.DataFrame(margin_rows), use_container_width=True, hide_index=True)
