# analyzer_from_csv_v2.py
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd


# =========================================================
# 0) CONFIG
# =========================================================

@dataclass
class AnalyzerConfig:
    data_dir: str = "."
    input_csv: str = "abbb.csv"

    # Period meta
    t1_start: str = "2025-11-01"
    t1_end: str = "2025-11-30"
    t2_start: str = "2025-12-01"
    t2_end: str = "2025-12-31"

    origin: str = "invoices"

    # Audit tolerancia (HUF) - csak számolásra (nem kell riportban)
    bridge_tolerance: int = 5000

    # CSV parsing
    csv_encoding: str = "cp1250"
    csv_encoding_errors: str = "strict"
    csv_engine: str = "python"
    csv_on_bad_lines: str = "skip"

    # Inventory snapshot input
    inventory_csv: Optional[str] = "BASE_TABLE_INVENTORY.csv"
    inventory_encoding: str = "cp1250"
    inventory_engine: str = "python"
    inventory_on_bad_lines: str = "skip"

    # Output
    output_json: str = "llm_input_v2.json"


# =========================================================
# 1) IO helpers
# =========================================================

USECOLS = [
    "created",
    "origin",
    "erp_invoice_id",
    "CT2_pack",
    "item_quantity",
    "revenues_wdisc_in_base_currency",
    "net_margin_wdisc_in_base_currency",
    "repeat_buyer",
    "user_id",
]

SAFE_DTYPES = {
    "origin": "category",
    "erp_invoice_id": "string",
    "CT2_pack": "string",
    "repeat_buyer": "string",
    "user_id": "string",
}

INVENTORY_USECOLS = [
    "CT2_pack",
    "stock_actual_quantity",
]


def _parse_period_bounds(cfg: AnalyzerConfig) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    t1s = pd.Timestamp(cfg.t1_start)
    t1e = pd.Timestamp(cfg.t1_end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    t2s = pd.Timestamp(cfg.t2_start)
    t2e = pd.Timestamp(cfg.t2_end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    return {"t1": (t1s, t1e), "t2": (t2s, t2e)}


def load_base_csv(cfg: AnalyzerConfig) -> pd.DataFrame:
    path = Path(cfg.data_dir) / cfg.input_csv
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path.resolve()}")

    df = pd.read_csv(
        path,
        usecols=lambda c: c in USECOLS,
        dtype={k: v for k, v in SAFE_DTYPES.items() if k in USECOLS and k != "created"},
        parse_dates=["created"],
        encoding=cfg.csv_encoding,
        encoding_errors=cfg.csv_encoding_errors,
        sep=",",
        quotechar='"',
        doublequote=True,
        engine=cfg.csv_engine,
        on_bad_lines=cfg.csv_on_bad_lines,
    )

    required = {
        "created", "origin", "erp_invoice_id", "CT2_pack",
        "item_quantity", "revenues_wdisc_in_base_currency"
    }
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["item_quantity"] = pd.to_numeric(df["item_quantity"], errors="coerce")
    df["revenues_wdisc_in_base_currency"] = pd.to_numeric(df["revenues_wdisc_in_base_currency"], errors="coerce")

    if "net_margin_wdisc_in_base_currency" in df.columns:
        df["net_margin_wdisc_in_base_currency"] = pd.to_numeric(df["net_margin_wdisc_in_base_currency"], errors="coerce")
    else:
        df["net_margin_wdisc_in_base_currency"] = np.nan

    return df


def load_inventory_csv(cfg: AnalyzerConfig) -> Optional[pd.DataFrame]:
    if not cfg.inventory_csv:
        return None

    path = Path(cfg.data_dir) / cfg.inventory_csv
    if not path.exists():
        raise FileNotFoundError(f"Missing inventory CSV: {path.resolve()}")

    inv = pd.read_csv(
        path,
        usecols=lambda c: c in INVENTORY_USECOLS,
        encoding=cfg.inventory_encoding,
        engine=cfg.inventory_engine,
        on_bad_lines=cfg.inventory_on_bad_lines,
    )

    if "CT2_pack" not in inv.columns:
        raise ValueError("Inventory CSV must contain CT2_pack")

    inv["CT2_pack"] = inv["CT2_pack"].astype("string").fillna("")
    inv["stock_actual_quantity"] = pd.to_numeric(inv["stock_actual_quantity"], errors="coerce").fillna(0.0)

    agg = (
        inv.groupby("CT2_pack", dropna=False)
        .agg(on_hand_qty=("stock_actual_quantity", "sum"))
        .reset_index()
    )

    agg["on_hand_qty"] = agg["on_hand_qty"].round(0)
    agg["in_stock_flag"] = (agg["on_hand_qty"] > 0).astype(int)

    return agg


def attach_inventory(df2w: pd.DataFrame, inv_pack: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = df2w.copy()
    if inv_pack is None or inv_pack.empty:
        out["on_hand_qty"] = np.nan
        out["in_stock_flag"] = np.nan
        return out
    return out.merge(inv_pack, on="CT2_pack", how="left")


# =========================================================
# 2) Core computations
# =========================================================

def filter_two_weeks(df: pd.DataFrame, cfg: AnalyzerConfig) -> pd.DataFrame:
    bounds = _parse_period_bounds(cfg)
    t1s, t1e = bounds["t1"]
    t2s, t2e = bounds["t2"]

    m_origin = (df["origin"].astype(str) == cfg.origin)
    m_t1 = (df["created"] >= t1s) & (df["created"] <= t1e)
    m_t2 = (df["created"] >= t2s) & (df["created"] <= t2e)

    out = df.loc[m_origin & (m_t1 | m_t2)].copy()
    out = out[out["item_quantity"].notna() & out["revenues_wdisc_in_base_currency"].notna()]

    rb = out["repeat_buyer"].astype("string")
    rb = rb.fillna("one_time").replace({"": "one_time", "nan": "one_time", "<NA>": "one_time"})
    out["repeat_buyer"] = rb

    return out


def add_period_flag(df: pd.DataFrame, cfg: AnalyzerConfig) -> pd.DataFrame:
    bounds = _parse_period_bounds(cfg)
    t1s, t1e = bounds["t1"]
    t2s, t2e = bounds["t2"]

    period = np.where(
        (df["created"] >= t1s) & (df["created"] <= t1e),
        "t1",
        np.where((df["created"] >= t2s) & (df["created"] <= t2e), "t2", None),
    )

    out = df.copy()
    out["period"] = period
    out = out[out["period"].notna()]
    return out


def compute_ct2_pack_weekly(df2w: pd.DataFrame) -> pd.DataFrame:
    for c in ["on_hand_qty", "in_stock_flag"]:
        if c not in df2w.columns:
            df2w[c] = np.nan

    g = (
        df2w.groupby(["CT2_pack", "period"], dropna=False)
        .agg(
            qty=("item_quantity", "sum"),
            rev=("revenues_wdisc_in_base_currency", "sum"),
            margin=("net_margin_wdisc_in_base_currency", "sum"),
            on_hand_qty=("on_hand_qty", "max"),
            in_stock_flag=("in_stock_flag", "max"),
        )
        .reset_index()
    )

    pv_qty = g.pivot(index="CT2_pack", columns="period", values="qty")
    pv_rev = g.pivot(index="CT2_pack", columns="period", values="rev")
    pv_mar = g.pivot(index="CT2_pack", columns="period", values="margin")
    pv_onhand = g.pivot(index="CT2_pack", columns="period", values="on_hand_qty")
    pv_flag = g.pivot(index="CT2_pack", columns="period", values="in_stock_flag")

    idx = pv_qty.index

    out = pd.DataFrame(
        {
            "CT2_pack": idx.astype("string"),
            "q1": pv_qty.get("t1", pd.Series(0.0, index=idx)).fillna(0.0),
            "q2": pv_qty.get("t2", pd.Series(0.0, index=idx)).fillna(0.0),
            "r1": pv_rev.get("t1", pd.Series(0.0, index=idx)).fillna(0.0),
            "r2": pv_rev.get("t2", pd.Series(0.0, index=idx)).fillna(0.0),
            "m1": pv_mar.get("t1", pd.Series(0.0, index=idx)).fillna(0.0),
            "m2": pv_mar.get("t2", pd.Series(0.0, index=idx)).fillna(0.0),
            "on_hand1": pv_onhand.get("t1", pd.Series(np.nan, index=idx)),
            "on_hand2": pv_onhand.get("t2", pd.Series(np.nan, index=idx)),
            "in_stock1": pv_flag.get("t1", pd.Series(np.nan, index=idx)),
            "in_stock2": pv_flag.get("t2", pd.Series(np.nan, index=idx)),
        }
    ).reset_index(drop=True)

    out["p1"] = np.where(out["q1"] != 0, out["r1"] / out["q1"], np.nan)
    out["p2"] = np.where(out["q2"] != 0, out["r2"] / out["q2"], np.nan)

    out["m_pct1"] = np.where(out["r1"] != 0, out["m1"] / out["r1"], np.nan)
    out["m_pct2"] = np.where(out["r2"] != 0, out["m2"] / out["r2"], np.nan)

    return out


def compute_presence(ct2_pack_weekly: pd.DataFrame) -> pd.DataFrame:
    p = ct2_pack_weekly.copy()
    p["in_t1"] = (p["q1"] != 0) | (p["r1"] != 0)
    p["in_t2"] = (p["q2"] != 0) | (p["r2"] != 0)
    return p[["CT2_pack", "in_t1", "in_t2"]]


def compute_revenue_bridge_summary(ct2_pack_weekly: pd.DataFrame, presence: pd.DataFrame) -> pd.DataFrame:
    w = ct2_pack_weekly.merge(presence, on="CT2_pack", how="left")
    appearing_total = w.loc[(~w["in_t1"]) & (w["in_t2"]), "r2"].sum()
    disappearing_total = w.loc[(w["in_t1"]) & (~w["in_t2"]), "r1"].sum()
    common_mask = (w["in_t1"]) & (w["in_t2"])
    common_delta_total = (w.loc[common_mask, "r2"] - w.loc[common_mask, "r1"]).sum()
    total_delta_from_bridge = appearing_total - disappearing_total + common_delta_total

    return pd.DataFrame([{
        "appearing_total": float(np.round(appearing_total, 0)),
        "disappearing_total": float(np.round(disappearing_total, 0)),
        "common_delta_total": float(np.round(common_delta_total, 0)),
        "total_delta_from_bridge": float(np.round(total_delta_from_bridge, 0)),
    }])


def compute_global_totals_check(df2w: pd.DataFrame) -> pd.DataFrame:
    g = (
        df2w.groupby("period")["revenues_wdisc_in_base_currency"]
        .sum()
        .reset_index()
        .rename(columns={"revenues_wdisc_in_base_currency": "revenue"})
    )
    g = g.set_index("period").reindex(["t1", "t2"]).fillna(0.0).reset_index()
    g["revenue"] = g["revenue"].round(0)
    return g


def compute_common_pvm(ct2_pack_weekly: pd.DataFrame, presence: pd.DataFrame) -> pd.DataFrame:
    w = ct2_pack_weekly.merge(presence, on="CT2_pack", how="left")

    # 1) COMMON = bridge-hez igazodó (presence common)
    common = w.loc[(w["in_t1"]) & (w["in_t2"])].copy()

    # 2) PV-re csak az a pack jó, ahol van q mindkét oldalon -> p1/p2 értelmezhető
    priced = common.loc[
        (common["q1"] != 0) & (common["q2"] != 0) & common["p1"].notna() & common["p2"].notna()
    ].copy()

    # 3) Nem árazható common (q1==0 vagy q2==0 vagy p1/p2 NaN)
    unpriced = common.loc[~common.index.isin(priced.index)].copy()


    # Truth (bridge common ΔR)
    delta_common_revenue = float((common["r2"] - common["r1"]).sum())

    # Pack-szintű PV (csak priced univerzumon)
    volume_effect = float((priced["p1"] * (priced["q2"] - priced["q1"])).sum())
    price_effect = float((priced["q2"] * (priced["p2"] - priced["p1"])).sum())

    # Mix = maradék, ami mindent “kiegyenlít”, beleértve az unpriced ΔR-t is
    mix_effect = delta_common_revenue - volume_effect - price_effect

    # külön tételnek számold ki a gap-et (ez volt nálad pl. -338 617 Ft)
    unpriced_delta_revenue = float((unpriced["r2"] - unpriced["r1"]).sum())

    return pd.DataFrame([{
        "delta_common_revenue": float(np.round(delta_common_revenue, 0)),
        "volume_effect": float(np.round(volume_effect, 0)),
        "price_effect": float(np.round(price_effect, 0)),
        "mix_effect": float(np.round(mix_effect, 0)),

        # új diagnosztika / külön tétel alapja
        "unpriced_common_pack_cnt": int(len(unpriced)),
        "unpriced_common_delta_revenue": float(np.round(unpriced_delta_revenue, 0)),
    }])


def compute_pack_quadrant(ct2_pack_weekly: pd.DataFrame) -> pd.DataFrame:
    w = ct2_pack_weekly.copy()
    w["delta_qty"] = w["q2"] - w["q1"]
    w["delta_rev"] = w["r2"] - w["r1"]

    def quad(row) -> str:
        p1, p2, q1, q2 = row["p1"], row["p2"], row["q1"], row["q2"]
        if pd.isna(p1) or pd.isna(p2):
            return "Neutral / No change"
        if (p2 > p1) and (q2 > q1):
            return "D1: Price↑ Volume↑"
        if (p2 > p1) and (q2 < q1):
            return "D2: Price↑ Volume↓"
        if (p2 < p1) and (q2 > q1):
            return "D3: Price↓ Volume↑"
        if (p2 < p1) and (q2 < q1):
            return "D4: Price↓ Volume↓"
        return "Neutral / No change"

    w["quadrant"] = w.apply(quad, axis=1)
    return w[["CT2_pack", "q1", "q2", "r1", "r2", "p1", "p2", "delta_qty", "delta_rev", "quadrant"]]


def summarize_quadrants(pack_quadrant: pd.DataFrame) -> List[Dict[str, Any]]:
    out = (
        pack_quadrant.groupby("quadrant", dropna=False)
        .agg(
            pack_cnt=("CT2_pack", "count"),
            t1_qty=("q1", "sum"),
            t2_qty=("q2", "sum"),
            delta_qty=("delta_qty", "sum"),
            t1_rev=("r1", "sum"),
            t2_rev=("r2", "sum"),
            delta_rev=("delta_rev", "sum"),
        )
        .reset_index()
    )

    order = [
        "D1: Price↑ Volume↑",
        "D2: Price↑ Volume↓",
        "D3: Price↓ Volume↑",
        "D4: Price↓ Volume↓",
        "Neutral / No change",
    ]
    out["__ord"] = out["quadrant"].astype(str).map({k: i for i, k in enumerate(order)}).fillna(999).astype(int)
    out = out.sort_values("__ord").drop(columns="__ord")
    return out.to_dict("records")


def quadrant_members(pack_quadrant: pd.DataFrame, top_n: int = 80) -> Dict[str, List[str]]:
    res: Dict[str, List[str]] = {}
    for quad, g in pack_quadrant.groupby("quadrant"):
        packs = g["CT2_pack"].astype(str).tolist()
        res[str(quad)] = packs[:top_n]
    return res


def compute_basket_kpis_period(df2w: pd.DataFrame) -> pd.DataFrame:
    inv = (
        df2w.groupby(["period", "erp_invoice_id"], dropna=False)
        .agg(units=("item_quantity", "sum"))
        .reset_index()
    )
    k = (
        inv.groupby("period")
        .agg(invoice_cnt=("erp_invoice_id", "nunique"), units=("units", "sum"))
        .reset_index()
    )
    k["units_per_invoice"] = np.where(k["invoice_cnt"] > 0, k["units"] / k["invoice_cnt"], np.nan)
    k = k[["period", "invoice_cnt", "units_per_invoice"]].copy()
    k["units_per_invoice"] = k["units_per_invoice"].round(3)
    return k.set_index("period").reindex(["t1", "t2"]).reset_index()


def compute_buyer_split_period(df2w: pd.DataFrame) -> pd.DataFrame:
    inv = (
        df2w.groupby(["period", "repeat_buyer", "erp_invoice_id"], dropna=False)
        .agg(revenues=("revenues_wdisc_in_base_currency", "sum"),
             units=("item_quantity", "sum"))
        .reset_index()
    )
    out = (
        inv.groupby(["period", "repeat_buyer"], dropna=False)
        .agg(invoice_cnt=("erp_invoice_id", "nunique"),
             revenues=("revenues", "sum"))
        .reset_index()
    )
    out["revenues"] = out["revenues"].round(0)
    return out


def compute_volume_contrib_top_packs_by_segment(df2w: pd.DataFrame, top_n: int = 3) -> Dict[str, Any]:
    # 1) Pack-level totals to get GLOBAL p1 (from t1)
    g_pack = (
        df2w.groupby(["CT2_pack", "period"], dropna=False)
        .agg(qty=("item_quantity", "sum"),
             rev=("revenues_wdisc_in_base_currency", "sum"))
        .reset_index()
    )
    pv_qty_pack = g_pack.pivot_table(index="CT2_pack", columns="period", values="qty", fill_value=0.0)
    pv_rev_pack = g_pack.pivot_table(index="CT2_pack", columns="period", values="rev", fill_value=0.0)

    q1_tot = pv_qty_pack.get("t1", pd.Series(0.0, index=pv_qty_pack.index))
    q2_tot = pv_qty_pack.get("t2", pd.Series(0.0, index=pv_qty_pack.index))
    r1_tot = pv_rev_pack.get("t1", pd.Series(0.0, index=pv_rev_pack.index))

    # COMMON packs only (same universe as PVM common)
    common_mask = (q1_tot != 0) & (q2_tot != 0)
    common_packs = q1_tot.index[common_mask]

    p1_global = (r1_tot / q1_tot).replace([np.inf, -np.inf], np.nan)
    p1_global = p1_global.loc[common_packs].astype(float)

    # 2) Segment-level quantities (q1_seg, q2_seg)
    g_seg = (
        df2w.groupby(["repeat_buyer", "CT2_pack", "period"], dropna=False)
        .agg(qty=("item_quantity", "sum"))
        .reset_index()
    )
    pv_qty_seg = g_seg.pivot_table(
        index=["repeat_buyer", "CT2_pack"],
        columns="period",
        values="qty",
        fill_value=0.0
    )

    # Keep only COMMON packs (global universe)
    pv_qty_seg = pv_qty_seg.loc[pv_qty_seg.index.get_level_values("CT2_pack").isin(common_packs)]

    q1_seg = pv_qty_seg.get("t1", pd.Series(0.0, index=pv_qty_seg.index))
    q2_seg = pv_qty_seg.get("t2", pd.Series(0.0, index=pv_qty_seg.index))
    delta_q_seg = q2_seg - q1_seg

    # Attach p1_global(pack) to each (segment, pack)
    pack_index = pv_qty_seg.index.get_level_values("CT2_pack")
    p1_for_rows = pack_index.map(p1_global.to_dict())  # vectorized map
    p1_for_rows = pd.Series(p1_for_rows, index=pv_qty_seg.index, dtype="float64")

    vol_eff_seg = delta_q_seg * p1_for_rows

    tmp = pd.DataFrame({
        "repeat_buyer": pv_qty_seg.index.get_level_values(0).astype(str),
        "CT2_pack": pack_index.astype(str),
        "delta_qty": delta_q_seg.values,
        "volume_effect": vol_eff_seg.values,
    })

    # Keep non-zero effects only
    tmp = tmp[(tmp["volume_effect"].notna()) & (tmp["volume_effect"] != 0)].copy()

    res: Dict[str, Any] = {}
    for seg, s in tmp.groupby("repeat_buyer", dropna=False):
        seg_total = float(s["volume_effect"].sum())
        denom = abs(seg_total) if abs(seg_total) > 0 else np.nan

        # --- NEGATÍV TOP3 (legnegatívabb)
        s_neg = s[s["volume_effect"] < 0].sort_values("volume_effect", ascending=True).head(top_n).copy()

        # --- POZITÍV TOP3 (legpozitívabb)
        s_pos = s[s["volume_effect"] > 0].sort_values("volume_effect", ascending=False).head(top_n).copy()

        def _mk_list(ss: pd.DataFrame) -> List[Dict[str, Any]]:
            out_list = []
            for _, r in ss.iterrows():
                share = (abs(float(r["volume_effect"])) / denom * 100.0) if denom == denom else np.nan
                out_list.append({
                    "CT2_pack": str(r["CT2_pack"]),
                    "volume_effect": float(np.round(float(r["volume_effect"]), 0)),
                    "share": float(np.round(share, 1)) if share == share else None,
                })
            return out_list

        res[str(seg)] = {
            "segment_volume_effect_total": float(np.round(seg_total, 0)),
            "top_packs_neg": _mk_list(s_neg),
            "top_packs_pos": _mk_list(s_pos),
        }

    return {"by_segment": res}



def build_volume_explanation(
    basket_kpis: pd.DataFrame,
    buyer_split_period: pd.DataFrame,
    volume_pack_contrib: Dict[str, Any],
    total_volume_effect: float,
) -> Dict[str, Any]:
    volx: Dict[str, Any] = {"basket": {}, "buyer_split": {}, "volume_pack_contrib": {}, "totals": {}}

    # basket KPI-k
    if basket_kpis is not None and len(basket_kpis) > 0:
        p = basket_kpis.set_index("period")
        if "t1" in p.index and "t2" in p.index:
            t1i = int(p.loc["t1", "invoice_cnt"])
            t2i = int(p.loc["t2", "invoice_cnt"])
            t1u = float(p.loc["t1", "units_per_invoice"])
            t2u = float(p.loc["t2", "units_per_invoice"])

            # interpretation
            if abs(t2u - t1u) < 0.15:
                interp = "order_driven"
            else:
                interp = "basket_driven"

            volx["basket"] = {
                "t1": {"invoice_cnt": t1i, "units_per_invoice": t1u},
                "t2": {"invoice_cnt": t2i, "units_per_invoice": t2u},
                "delta": {"invoice_cnt": int(t2i - t1i), "units_per_invoice": round(t2u - t1u, 3)},
                "interpretation": interp,
            }

    by_seg = (volume_pack_contrib or {}).get("by_segment", {}) or {}

    # total Volume effect (common) – nevező share-hez
    total_vol = float(np.round(total_volume_effect, 0))
    denom = abs(total_vol) if abs(total_vol) > 0 else np.nan
    volx["totals"] = {"total_volume_effect": total_vol}

    # repeat vs one_time bontás
    if buyer_split_period is not None and len(buyer_split_period) > 0:
        for seg in buyer_split_period["repeat_buyer"].dropna().unique():
            seg_key = str(seg)
            sub = buyer_split_period[buyer_split_period["repeat_buyer"] == seg].set_index("period")

            if "t1" in sub.index and "t2" in sub.index:
                t1_orders = int(sub.loc["t1", "invoice_cnt"])
                t2_orders = int(sub.loc["t2", "invoice_cnt"])

                seg_vol_total = float((by_seg.get(seg_key, {}) or {}).get("segment_volume_effect_total", 0.0))
                seg_vol_total = float(np.round(seg_vol_total, 0))

                share_total = (abs(seg_vol_total) / denom * 100.0) if denom == denom else None
                share_total = float(np.round(share_total, 1)) if share_total is not None else None

                volx["buyer_split"][seg_key] = {
                    "t1_orders": t1_orders,
                    "t2_orders": t2_orders,
                    "delta_orders": int(t2_orders - t1_orders),
                    "volume_effect_total": seg_vol_total,
                    "share_of_total_volume_effect_pct": share_total,
                    "delta_revenue": seg_vol_total,  # compat
                }

    volx["volume_pack_contrib"] = by_seg
    return volx


def _compute_common_pack_price_table(df2w: pd.DataFrame) -> pd.DataFrame:
    """
    Pack-level p1/p2 (global, not by segment) and COMMON-pack universe.
    COMMON = pack has qty in both t1 and t2 (same universe as common PVM).
    Returns DataFrame with CT2_pack, p1_global, p2_global, is_common.
    """
    g_pack = (
        df2w.groupby(["CT2_pack", "period"], dropna=False)
        .agg(
            qty=("item_quantity", "sum"),
            rev=("revenues_wdisc_in_base_currency", "sum"),
        )
        .reset_index()
    )

    pv_qty = g_pack.pivot_table(index="CT2_pack", columns="period", values="qty", fill_value=0.0)
    pv_rev = g_pack.pivot_table(index="CT2_pack", columns="period", values="rev", fill_value=0.0)

    q1 = pv_qty.get("t1", pd.Series(0.0, index=pv_qty.index))
    q2 = pv_qty.get("t2", pd.Series(0.0, index=pv_qty.index))
    r1 = pv_rev.get("t1", pd.Series(0.0, index=pv_rev.index))
    r2 = pv_rev.get("t2", pd.Series(0.0, index=pv_rev.index))

    p1 = (r1 / q1).replace([np.inf, -np.inf], np.nan)
    p2 = (r2 / q2).replace([np.inf, -np.inf], np.nan)

    is_common = (q1 != 0) & (q2 != 0)

    out = pd.DataFrame({
        "CT2_pack": pv_qty.index.astype("string"),
        "p1_global": p1.astype(float).values,
        "p2_global": p2.astype(float).values,
        "is_common": is_common.astype(bool).values,
    })

    return out


def compute_price_contrib_top_packs_by_segment(df2w: pd.DataFrame, top_n: int = 3) -> Dict[str, Any]:
    """
    Price effect split by repeat_buyer, computed on COMMON packs only
    and using pack-level (global) p1/p2 (same universe + pricing as common PVM).

    price_effect_seg(pack) = q2_seg(pack) * (p2_global(pack) - p1_global(pack))
    """
    # 1) Global pack price table + common universe
    pack_price = _compute_common_pack_price_table(df2w)
    pack_price = pack_price[pack_price["is_common"]].copy()

    if pack_price.empty:
        return {"by_segment": {}}

    # dict maps for fast vectorized lookup
    p1_map = dict(zip(pack_price["CT2_pack"].astype(str), pack_price["p1_global"].astype(float)))
    p2_map = dict(zip(pack_price["CT2_pack"].astype(str), pack_price["p2_global"].astype(float)))
    common_packs = set(pack_price["CT2_pack"].astype(str).tolist())

    # 2) Segment-level quantities (q2_seg) by pack
    g = (
        df2w.groupby(["repeat_buyer", "CT2_pack", "period"], dropna=False)
        .agg(qty=("item_quantity", "sum"))
        .reset_index()
    )

    pv_qty = g.pivot_table(
        index=["repeat_buyer", "CT2_pack"],
        columns="period",
        values="qty",
        fill_value=0.0
    )

    # keep only common packs
    pv_qty = pv_qty.loc[pv_qty.index.get_level_values("CT2_pack").astype(str).isin(common_packs)]

    idx = pv_qty.index
    q2_seg = pv_qty.get("t2", pd.Series(0.0, index=idx))

    # 3) Attach global p1/p2 per pack to each (segment, pack)
    pack_idx = idx.get_level_values("CT2_pack").astype(str)
    p1_global = pack_idx.map(p1_map)
    p2_global = pack_idx.map(p2_map)

    p1_global = pd.Series(p1_global, index=idx, dtype="float64")
    p2_global = pd.Series(p2_global, index=idx, dtype="float64")

    # 4) Compute price effect per (segment, pack)
    price_eff = q2_seg.astype(float) * (p2_global - p1_global)

    tmp = pd.DataFrame({
        "repeat_buyer": idx.get_level_values(0).astype(str),
        "CT2_pack": pack_idx,
        "q2_seg": q2_seg.values,
        "price_effect": price_eff.values,
    })

    # keep valid rows only
    tmp = tmp[
        (tmp["q2_seg"] != 0) &
        (tmp["price_effect"].notna()) &
        (tmp["price_effect"] != 0)
    ].copy()

        # 5) Aggregate + TopN per segment (neg / pos)
    res: Dict[str, Any] = {}
    for seg, s in tmp.groupby("repeat_buyer", dropna=False):
        seg_total = float(s["price_effect"].sum())   # <-- EZ volt rossz (volume_effect)
        denom = abs(seg_total) if abs(seg_total) > 0 else np.nan

        def _mk_top_list(ss: pd.DataFrame) -> List[Dict[str, Any]]:
            out_list = []
            for _, r in ss.iterrows():
                share = (abs(float(r["price_effect"])) / denom * 100.0) if denom == denom else np.nan
                out_list.append({
                    "CT2_pack": str(r["CT2_pack"]),
                    "price_effect": float(np.round(float(r["price_effect"]), 0)),
                    "share": float(np.round(share, 1)) if share == share else None,
                })
            return out_list

        neg = s.sort_values("price_effect", ascending=True).head(top_n).copy()
        pos = s.sort_values("price_effect", ascending=False).head(top_n).copy()

        res[str(seg)] = {
            "segment_price_effect_total": float(np.round(seg_total, 0)),
            "top_packs_neg": _mk_top_list(neg),
            "top_packs_pos": _mk_top_list(pos),
        }

    return {"by_segment": res}




def build_price_explanation(buyer_split_period: pd.DataFrame, price_pack_contrib: Dict[str, Any]) -> Dict[str, Any]:
    px: Dict[str, Any] = {"buyer_split": {}, "price_pack_contrib": {}}
    if buyer_split_period is not None and len(buyer_split_period) > 0:
        for seg in buyer_split_period["repeat_buyer"].dropna().unique():
            px["buyer_split"][str(seg)] = {"present": True}
    px["price_pack_contrib"] = price_pack_contrib.get("by_segment", {})
    return px


def compute_quadrant_revenue_contrib_top(df2w: pd.DataFrame, top_n: int = 20) -> Dict[str, Any]:
    g = (
        df2w.groupby(["CT2_pack", "period"], dropna=False)
        .agg(qty=("item_quantity", "sum"),
             rev=("revenues_wdisc_in_base_currency", "sum"))
        .reset_index()
    )

    pv_qty = g.pivot_table(index="CT2_pack", columns="period", values="qty", fill_value=0.0)
    pv_rev = g.pivot_table(index="CT2_pack", columns="period", values="rev", fill_value=0.0)

    base = pd.DataFrame({
        "CT2_pack": pv_qty.index.astype("string"),
        "q1": pv_qty.get("t1", pd.Series(0.0, index=pv_qty.index)).values,
        "q2": pv_qty.get("t2", pd.Series(0.0, index=pv_qty.index)).values,
        "r1": pv_rev.get("t1", pd.Series(0.0, index=pv_rev.index)).values,
        "r2": pv_rev.get("t2", pd.Series(0.0, index=pv_rev.index)).values,
    })

    base["p1"] = np.where(base["q1"] != 0, base["r1"] / base["q1"], np.nan)
    base["p2"] = np.where(base["q2"] != 0, base["r2"] / base["q2"], np.nan)
    base["delta_rev"] = base["r2"] - base["r1"]

    def quad_row(row) -> str:
        p1, p2, q1, q2 = row["p1"], row["p2"], row["q1"], row["q2"]
        if pd.isna(p1) or pd.isna(p2):
            return "Neutral / No change"
        if (p2 > p1) and (q2 > q1):
            return "D1: Price↑ Volume↑"
        if (p2 > p1) and (q2 < q1):
            return "D2: Price↑ Volume↓"
        if (p2 < p1) and (q2 > q1):
            return "D3: Price↓ Volume↑"
        if (p2 < p1) and (q2 < q1):
            return "D4: Price↓ Volume↓"
        return "Neutral / No change"

    
    base["quadrant"] = base.apply(quad_row, axis=1)

    res: Dict[str, Any] = {}
    for quad, s in base.groupby("quadrant", dropna=False):
        quad_total = float(s["delta_rev"].sum())
        denom = abs(quad_total) if abs(quad_total) > 0 else np.nan

        s2 = s.copy()
        s2["abs_delta_rev"] = s2["delta_rev"].abs()
        s2 = s2.sort_values("abs_delta_rev", ascending=False).head(top_n)

        items = []
        for _, r in s2.iterrows():
            share = (abs(float(r["delta_rev"])) / denom * 100.0) if denom == denom else np.nan
            items.append({
                "CT2_pack": str(r["CT2_pack"]),
                "delta_rev": float(np.round(float(r["delta_rev"]), 0)),
                "share_pct": float(np.round(share, 1)) if share == share else None,
            })

        res[str(quad)] = {
            "quadrant_delta_rev_total": float(np.round(quad_total, 0)),
            "top_products": items,
        }

    return res


def compute_quadrant_key_products(df2w: pd.DataFrame, top_n: int = 6, coverage: float = 0.7) -> Dict[str, Any]:
    """
    Per quadrant key products:
    - Build COMMON pack universe (qty in both t1 and t2) for PV/Impact calculation
    - Impact = price_effect + volume_effect
    - Margin delta = m2 - m1
    Selection:
    - rank by abs(impact)
    - take until coverage of abs(total impact) OR top_n
    Output: dict[quadrant] = {"items":[...], "coverage_pct":...}
    """

    g = (
        df2w.groupby(["CT2_pack", "period"], dropna=False)
        .agg(
            qty=("item_quantity", "sum"),
            rev=("revenues_wdisc_in_base_currency", "sum"),
            mar=("net_margin_wdisc_in_base_currency", "sum"),
        )
        .reset_index()
    )

    pv_qty = g.pivot_table(index="CT2_pack", columns="period", values="qty", fill_value=0.0)
    pv_rev = g.pivot_table(index="CT2_pack", columns="period", values="rev", fill_value=0.0)
    pv_mar = g.pivot_table(index="CT2_pack", columns="period", values="mar", fill_value=0.0)

    base = pd.DataFrame({
        "CT2_pack": pv_qty.index.astype("string"),
        "q1": pv_qty.get("t1", pd.Series(0.0, index=pv_qty.index)).values,
        "q2": pv_qty.get("t2", pd.Series(0.0, index=pv_qty.index)).values,
        "r1": pv_rev.get("t1", pd.Series(0.0, index=pv_rev.index)).values,
        "r2": pv_rev.get("t2", pd.Series(0.0, index=pv_rev.index)).values,
        "m1": pv_mar.get("t1", pd.Series(0.0, index=pv_mar.index)).values,
        "m2": pv_mar.get("t2", pd.Series(0.0, index=pv_mar.index)).values,
    })

    base["p1"] = np.where(base["q1"] != 0, base["r1"] / base["q1"], np.nan)
    base["p2"] = np.where(base["q2"] != 0, base["r2"] / base["q2"], np.nan)

    # COMMON packs only
    base = base[(base["q1"] != 0) & (base["q2"] != 0) & base["p1"].notna() & base["p2"].notna()].copy()

    base["volume_effect"] = base["p1"] * (base["q2"] - base["q1"])
    base["price_effect"] = base["q2"] * (base["p2"] - base["p1"])
    base["impact_effect"] = base["volume_effect"] + base["price_effect"]
    base["delta_margin"] = base["m2"] - base["m1"]

    # quadrant by sign(Price, Volume)
    def quad_row(row) -> str:
        pe, ve = row["price_effect"], row["volume_effect"]
        if pd.isna(pe) or pd.isna(ve):
            return "Neutral / No change"
        if (pe > 0) and (ve > 0):
            return "D1: Price↑ Volume↑"
        if (pe > 0) and (ve < 0):
            return "D2: Price↑ Volume↓"
        if (pe < 0) and (ve > 0):
            return "D3: Price↓ Volume↑"
        if (pe < 0) and (ve < 0):
            return "D4: Price↓ Volume↓"
        return "Neutral / No change"

    base["quadrant"] = base.apply(quad_row, axis=1)

    # aggregate per quadrant
    out: Dict[str, Any] = {}
    for quad, s in base.groupby("quadrant", dropna=False):
        total_abs = float(s["impact_effect"].abs().sum())
        s2 = s.copy()
        s2["abs_impact"] = s2["impact_effect"].abs()
        s2 = s2.sort_values("abs_impact", ascending=False)

        items = []
        running = 0.0
        for _, r in s2.iterrows():
            if len(items) >= top_n:
                break
            items.append({
                "CT2_pack": str(r["CT2_pack"]),
                "price_effect": float(np.round(float(r["price_effect"]), 0)),
                "volume_effect": float(np.round(float(r["volume_effect"]), 0)),
                "impact_effect": float(np.round(float(r["impact_effect"]), 0)),
                "delta_margin": float(np.round(float(r["delta_margin"]), 0)),
            })
            running += abs(float(r["impact_effect"]))
            if total_abs > 0 and (running / total_abs) >= coverage:
                break

        cov = (running / total_abs * 100.0) if total_abs > 0 else None
        out[str(quad)] = {
            "items": items,
            "coverage_pct": float(np.round(cov, 1)) if cov is not None else None
        }

    return out

def compute_quadrant_impact_contrib_top(df2w: pd.DataFrame, top_n: int = 20) -> Dict[str, Any]:
    """
    Quadrant contribution (A1 + residual):
    - Quadrant label: pack-szintű jel (priced packeknél) + unpriced -> Neutral / No change
    - Quadrant TOTALS: "pack-konzisztens" PV/Impact aggregáció (sum of pack-level PV on priced packs)
        volume_total = Σ[p1_pack * (q2_pack - q1_pack)]  (priced packek)
        price_total  = Σ[q2_pack * (p2_pack - p1_pack)]  (priced packek)
        impact_total = volume_total + price_total
        delta_rev_total = Σ(r2 - r1)  (priced + unpriced common is benne)
      Kvadráns-szinten NEM riportolunk mix_total-t.
    - Residual (reconciliation): residual_mix_total = common_delta_rev_total - Σ(quadrant_volume_total) - Σ(quadrant_price_total)
      -> ennek egyeznie kell a common_pvm.mix_effect-tel (kerekítési eltéréssel).
    - Top products: pack-szintű impact alapján (Price_pack + Volume_pack)
    """

    print("DEBUG: quadrant_impact_contrib_top VERSION = 2026-01-19-A1-RESIDUAL")

    # -------------------------
    # 1) Pack totals (t1/t2)
    # -------------------------
    g = (
        df2w.groupby(["CT2_pack", "period"], dropna=False)
        .agg(
            qty=("item_quantity", "sum"),
            rev=("revenues_wdisc_in_base_currency", "sum"),
        )
        .reset_index()
    )

    pv_qty = g.pivot_table(index="CT2_pack", columns="period", values="qty", fill_value=0.0)
    pv_rev = g.pivot_table(index="CT2_pack", columns="period", values="rev", fill_value=0.0)

    base = pd.DataFrame({
        "CT2_pack": pv_qty.index.astype("string"),
        "q1": pv_qty.get("t1", pd.Series(0.0, index=pv_qty.index)).values,
        "q2": pv_qty.get("t2", pd.Series(0.0, index=pv_qty.index)).values,
        "r1": pv_rev.get("t1", pd.Series(0.0, index=pv_rev.index)).values,
        "r2": pv_rev.get("t2", pd.Series(0.0, index=pv_rev.index)).values,
    })

    # unit prices (pack-level)
    base["p1"] = np.where(base["q1"] != 0, base["r1"] / base["q1"], np.nan)
    base["p2"] = np.where(base["q2"] != 0, base["r2"] / base["q2"], np.nan)

    # -------------------------
    # 2) COMMON by PRESENCE (bridge-compatible)
    # -------------------------
    base["in_t1"] = (base["q1"] != 0) | (base["r1"] != 0)
    base["in_t2"] = (base["q2"] != 0) | (base["r2"] != 0)
    base = base[base["in_t1"] & base["in_t2"]].copy()

    base["delta_qty"] = base["q2"] - base["q1"]
    base["delta_rev"] = base["r2"] - base["r1"]

    # priced = tudunk p1/p2-t értelmezni (q1,q2 != 0 és p1,p2 nem NaN)
    base["is_priced"] = (base["q1"] != 0) & (base["q2"] != 0) & base["p1"].notna() & base["p2"].notna()
    pm = base["is_priced"]

    # -------------------------
    # 3) Pack-level PV (csak priced packekre)
    # -------------------------
    base["volume_effect_pack"] = 0.0
    base["price_effect_pack"] = 0.0

    base.loc[pm, "volume_effect_pack"] = base.loc[pm, "p1"] * (base.loc[pm, "q2"] - base.loc[pm, "q1"])
    base.loc[pm, "price_effect_pack"]  = base.loc[pm, "q2"] * (base.loc[pm, "p2"] - base.loc[pm, "p1"])

    base["impact_effect_pack"] = base["volume_effect_pack"] + base["price_effect_pack"]

    # -------------------------
    # 4) Quadrant label (A1)
    #    - unpriced common -> Neutral / No change
    #    - priced -> D1..D4 jel a pack-szintű PV előjelekből
    # -------------------------
    def quad_row(row) -> str:
        if not bool(row["is_priced"]):
            return "Neutral / No change"
        pe, ve = row["price_effect_pack"], row["volume_effect_pack"]
        if (pe > 0) and (ve > 0):
            return "D1: Price↑ Volume↑"
        if (pe > 0) and (ve < 0):
            return "D2: Price↑ Volume↓"
        if (pe < 0) and (ve > 0):
            return "D3: Price↓ Volume↑"
        if (pe < 0) and (ve < 0):
            return "D4: Price↓ Volume↓"
        return "Neutral / No change"

    base["quadrant"] = base.apply(quad_row, axis=1)

    # -------------------------
    # 5) Neutral threshold (csak priced packekre, impact_pack alapján)
    # -------------------------
    total_abs_impact = float(base.loc[pm, "impact_effect_pack"].abs().sum())
    thr = max(10_000.0, 0.001 * total_abs_impact) if total_abs_impact > 0 else 10_000.0
    neutral_mask = pm & (base["impact_effect_pack"].abs() < thr)
    base.loc[neutral_mask, "quadrant"] = "Neutral / No change"

    # -------------------------
    # 6) Kvadráns-aggregáció (A1): vol/price/impact összegek PACK-SUM módon
    #    - delta_rev_total viszont tartalmazza az unpriced common ΔR-t is (Neutralban jellemzően)
    # -------------------------
    res: Dict[str, Any] = {}

    for quad, s in base.groupby("quadrant", dropna=False):
        q_volume = float(s.loc[s["is_priced"], "volume_effect_pack"].sum())
        q_price  = float(s.loc[s["is_priced"], "price_effect_pack"].sum())
        q_impact = float(q_volume + q_price)

        dR = float(s["delta_rev"].sum())  # priced + unpriced

        # TOP list (diagnosztika): abs(pack impact)
        s2 = s.copy()
        s2["abs_rank"] = s2["impact_effect_pack"].abs()
        s2 = s2.sort_values("abs_rank", ascending=False).head(top_n)

        denom = abs(q_impact) if abs(q_impact) > 0 else np.nan
        items = []
        for _, r in s2.iterrows():
            share = (abs(float(r["impact_effect_pack"])) / denom * 100.0) if denom == denom else np.nan
            items.append({
                "CT2_pack": str(r["CT2_pack"]),
                "impact_effect": float(np.round(float(r["impact_effect_pack"]), 0)),
                "share_pct": float(np.round(share, 1)) if share == share else None,
                "price_effect": float(np.round(float(r["price_effect_pack"]), 0)),
                "volume_effect": float(np.round(float(r["volume_effect_pack"]), 0)),
                "delta_rev": float(np.round(float(r["delta_rev"]), 0)),
                "is_priced": bool(r["is_priced"]),
            })

        res[str(quad)] = {
            "pack_cnt": int(len(s)),
            "delta_qty_total": float(np.round(float(s["delta_qty"].sum()), 0)),

            # A1: nincs mix_total kvadránsonként
            "impact_total": float(np.round(q_impact, 0)),
            "price_total":  float(np.round(q_price, 0)),
            "volume_total": float(np.round(q_volume, 0)),
            "delta_rev_total": float(np.round(dR, 0)),

            "top_products": items,
            "neutral_threshold": float(np.round(thr, 0)),
        }

    # -------------------------
    # 7) Reconciliation (residual mix) – hogy egyezzen a common PVM-mel
    # -------------------------
    # common ΔR (presence-common univerzum)
    common_delta_rev_total = float(base["delta_rev"].sum())

    # Σ vol/price a kvadránsokból (csak priced PV összegek)
    sum_volume = 0.0
    sum_price = 0.0
    for k, v in res.items():
        if k == "_reconciliation":
            continue
        sum_volume += float(v.get("volume_total", 0.0))
        sum_price += float(v.get("price_total", 0.0))

    residual_mix = float(common_delta_rev_total - sum_volume - sum_price)

    res["_reconciliation"] = {
        "common_delta_rev_total": float(np.round(common_delta_rev_total, 0)),
        "sum_quadrant_volume_total": float(np.round(sum_volume, 0)),
        "sum_quadrant_price_total": float(np.round(sum_price, 0)),
        "residual_mix_total": float(np.round(residual_mix, 0)),

        # opcionális debug – ha kell
        "unpriced_common_pack_cnt": int((~base["is_priced"]).sum()),
        "unpriced_common_delta_rev_total": float(np.round(float(base.loc[~base["is_priced"], "delta_rev"].sum()), 0)),
    }

    return res

        


def compute_margin_kpis(df2w: pd.DataFrame) -> Dict[str, Any]:
    g_rev = df2w.groupby("period")["revenues_wdisc_in_base_currency"].sum()
    g_mar = df2w.groupby("period")["net_margin_wdisc_in_base_currency"].sum()

    t1m = float(g_mar.get("t1", 0.0))
    t2m = float(g_mar.get("t2", 0.0))
    t1r = float(g_rev.get("t1", 0.0))
    t2r = float(g_rev.get("t2", 0.0))

    return {
        "t1_margin": float(np.round(t1m, 0)),
        "t2_margin": float(np.round(t2m, 0)),
        "delta_margin": float(np.round(t2m - t1m, 0)),
        "t1_margin_pct": round((t1m / t1r), 4) if t1r else None,
        "t2_margin_pct": round((t2m / t2r), 4) if t2r else None,
    }


# =========================================================
# 3) Build llm_input_v2 JSON
# =========================================================

def compute_global_kpis(global_totals_check: pd.DataFrame) -> Dict[str, Any]:
    p = global_totals_check.set_index("period")
    t1_rev = float(p.loc["t1", "revenue"])
    t2_rev = float(p.loc["t2", "revenue"])
    return {"t1_revenue": t1_rev, "t2_revenue": t2_rev, "delta_revenue": float(t2_rev - t1_rev)}


def compute_math_check(global_kpis: Dict[str, Any], revenue_bridge_summary: pd.DataFrame, tol: int) -> Dict[str, Any]:
    row = revenue_bridge_summary.iloc[0].to_dict()
    bridge_delta = float(row["total_delta_from_bridge"])
    global_delta = float(global_kpis["delta_revenue"])
    diff = float(bridge_delta - global_delta)
    return {
        "audit_passes": bool(abs(diff) <= tol),
        "bridge_vs_global_diff": float(np.round(diff, 0)),
        "tolerance": int(tol),
    }

def compute_quadrant_margin_contrib_top(df2w: pd.DataFrame, top_n: int = 20) -> Dict[str, Any]:
    """
    Quadrant margin diagnostic (C style):
    - quadrant_delta_margin_total = Σ(m2 - m1)
    - top_improver: max +delta_margin pack
    - top_detractor: min -delta_margin pack
    - (optional) top_n lists by abs(delta_margin) if you ever need later
    """
    g = (
        df2w.groupby(["CT2_pack", "period"], dropna=False)
        .agg(
            qty=("item_quantity", "sum"),
            rev=("revenues_wdisc_in_base_currency", "sum"),
            mar=("net_margin_wdisc_in_base_currency", "sum"),
        )
        .reset_index()
    )

    pv_qty = g.pivot_table(index="CT2_pack", columns="period", values="qty", fill_value=0.0)
    pv_rev = g.pivot_table(index="CT2_pack", columns="period", values="rev", fill_value=0.0)
    pv_mar = g.pivot_table(index="CT2_pack", columns="period", values="mar", fill_value=0.0)

    base = pd.DataFrame({
        "CT2_pack": pv_qty.index.astype("string"),
        "q1": pv_qty.get("t1", pd.Series(0.0, index=pv_qty.index)).values,
        "q2": pv_qty.get("t2", pd.Series(0.0, index=pv_qty.index)).values,
        "r1": pv_rev.get("t1", pd.Series(0.0, index=pv_rev.index)).values,
        "r2": pv_rev.get("t2", pd.Series(0.0, index=pv_rev.index)).values,
        "m1": pv_mar.get("t1", pd.Series(0.0, index=pv_mar.index)).values,
        "m2": pv_mar.get("t2", pd.Series(0.0, index=pv_mar.index)).values,
    })

    base["p1"] = np.where(base["q1"] != 0, base["r1"] / base["q1"], np.nan)
    base["p2"] = np.where(base["q2"] != 0, base["r2"] / base["q2"], np.nan)

    # ugyanaz a kvadráns logika, mint nálad máshol (price & volume irány)
    def quad_row(row) -> str:
        p1, p2, q1, q2 = row["p1"], row["p2"], row["q1"], row["q2"]
        if pd.isna(p1) or pd.isna(p2):
            return "Neutral / No change"
        if (p2 > p1) and (q2 > q1):
            return "D1: Price↑ Volume↑"
        if (p2 > p1) and (q2 < q1):
            return "D2: Price↑ Volume↓"
        if (p2 < p1) and (q2 > q1):
            return "D3: Price↓ Volume↑"
        if (p2 < p1) and (q2 < q1):
            return "D4: Price↓ Volume↓"
        return "Neutral / No change"

    base["quadrant"] = base.apply(quad_row, axis=1)

    # margin delta pack szinten
    base["delta_margin"] = base["m2"] - base["m1"]

    res: Dict[str, Any] = {}

    for quad, s in base.groupby("quadrant", dropna=False):
        q_total = float(s["delta_margin"].sum())
        pack_cnt = int(len(s))

        # improver = legnagyobb pozitív
        s_pos = s[s["delta_margin"] > 0].sort_values("delta_margin", ascending=False)
        top_improver = None
        if len(s_pos) > 0:
            r = s_pos.iloc[0]
            top_improver = {
                "CT2_pack": str(r["CT2_pack"]),
                "delta_margin": float(np.round(float(r["delta_margin"]), 0)),
            }

        # detractor = legnegatívabb
        s_neg = s[s["delta_margin"] < 0].sort_values("delta_margin", ascending=True)
        top_detractor = None
        if len(s_neg) > 0:
            r = s_neg.iloc[0]
            top_detractor = {
                "CT2_pack": str(r["CT2_pack"]),
                "delta_margin": float(np.round(float(r["delta_margin"]), 0)),
            }

        # (opcionális) top_n abs szerinti lista - nem muszáj renderelni most
        s_abs = s.copy()
        s_abs["abs_delta_margin"] = s_abs["delta_margin"].abs()
        s_abs = s_abs.sort_values("abs_delta_margin", ascending=False).head(top_n)
        top_by_abs = [
            {"CT2_pack": str(r["CT2_pack"]), "delta_margin": float(np.round(float(r["delta_margin"]), 0))}
            for _, r in s_abs.iterrows()
            if float(r["delta_margin"]) != 0
        ]

        res[str(quad)] = {
            "pack_cnt": pack_cnt,
            "quadrant_delta_margin_total": float(np.round(q_total, 0)),
            "top_improver": top_improver,
            "top_detractor": top_detractor,
            "top_products_by_abs": top_by_abs,  # ha később kell
        }

    return res



def build_llm_input_v2(cfg: AnalyzerConfig) -> Dict[str, Any]:
    df = load_base_csv(cfg)
    df2w = filter_two_weeks(df, cfg)
    df2w = add_period_flag(df2w, cfg)

    inv_pack = load_inventory_csv(cfg)
    df2w = attach_inventory(df2w, inv_pack)

    ct2_pack_weekly = compute_ct2_pack_weekly(df2w)
    presence = compute_presence(ct2_pack_weekly)

    revenue_bridge_summary = compute_revenue_bridge_summary(ct2_pack_weekly, presence)
    global_totals_check = compute_global_totals_check(df2w)
    common_pvm = compute_common_pvm(ct2_pack_weekly, presence)

    pack_quadrant = compute_pack_quadrant(ct2_pack_weekly)
    quads_summary = summarize_quadrants(pack_quadrant)
    quads_members = quadrant_members(pack_quadrant, top_n=80)

    basket_kpis_period = compute_basket_kpis_period(df2w)
    buyer_split_period = compute_buyer_split_period(df2w)

    volume_pack_contrib = compute_volume_contrib_top_packs_by_segment(df2w, top_n=3)
    pvm_row = common_pvm.iloc[0].to_dict()
    total_volume_effect = float(pvm_row.get("volume_effect", 0.0))
    volume_explanation = build_volume_explanation(
    basket_kpis_period,
    buyer_split_period,
    volume_pack_contrib,
    total_volume_effect=total_volume_effect
)

    price_pack_contrib = compute_price_contrib_top_packs_by_segment(df2w, top_n=3)
    price_explanation = build_price_explanation(buyer_split_period, price_pack_contrib)

    quadrant_impact_contrib = compute_quadrant_impact_contrib_top(df2w, top_n=20)
    quadrant_margin_contrib = compute_quadrant_margin_contrib_top(df2w, top_n=20)
    quadrant_key_products = compute_quadrant_key_products(df2w, top_n=6, coverage=0.7)


    global_kpis = compute_global_kpis(global_totals_check)
    math_check = compute_math_check(global_kpis, revenue_bridge_summary, cfg.bridge_tolerance)

    b = revenue_bridge_summary.iloc[0].to_dict()
    bridge_dict = {
        "appearing_total": float(b["appearing_total"]),
        "disappearing_total": float(b["disappearing_total"]),
        "common_delta_total": float(b["common_delta_total"]),
        "total_delta_from_bridge": float(b["total_delta_from_bridge"]),
    }

    pvm_row = common_pvm.iloc[0].to_dict()
    common_pvm_dict = {
        "delta_common_revenue": float(pvm_row.get("delta_common_revenue", 0.0)),
        "volume_effect": float(pvm_row.get("volume_effect", 0.0)),
        "price_effect": float(pvm_row.get("price_effect", 0.0)),
        "mix_effect": float(pvm_row.get("mix_effect", 0.0)),
    }

    inventory_kpis = {}
    if inv_pack is not None and len(inv_pack) > 0:
        inventory_kpis = {
            "packs_total": int(len(inv_pack)),
            "packs_in_stock": int((inv_pack["in_stock_flag"] == 1).sum()),
            "total_on_hand_qty": float(np.round(inv_pack["on_hand_qty"].sum(), 0)),
        }

    margin_kpis = compute_margin_kpis(df2w)

    llm_input_v2 = {
        "periods": {
            "t1": {"start": cfg.t1_start, "end": cfg.t1_end},
            "t2": {"start": cfg.t2_start, "end": cfg.t2_end},
        },
        "global_kpis": global_kpis,
        "bridge": bridge_dict,
        "math_check": math_check,
        "common_pvm": common_pvm_dict,
        "quadrants": {"summary": quads_summary, "members_ct2_pack": quads_members},

        "volume_explanation": volume_explanation,
        "price_explanation": price_explanation,
        "quadrant_impact_contrib": quadrant_impact_contrib,
        "quadrant_margin_contrib": quadrant_margin_contrib,
        "quadrant_key_products": quadrant_key_products,

        "margin_kpis": margin_kpis,

        "inventory_kpis": inventory_kpis,
        "inventory_meta": {
            "source": cfg.inventory_csv,
            "note": "Inventory is a snapshot joined by CT2_pack (no history). Use only for action buckets.",
            "fields": ["on_hand_qty", "in_stock_flag"],
        },

        "_meta": {
            "source": f"CSV: {cfg.input_csv}",
            "filters": {"origin": cfg.origin, "t1": [cfg.t1_start, cfg.t1_end], "t2": [cfg.t2_start, cfg.t2_end]},
            "notes": [
                "All computed in pandas from the raw export.",
                "PVM computed on COMMON packs only; appearing/disappearing handled in bridge.",
            ],
        },
    }

    return llm_input_v2


def run_analyzer(cfg: AnalyzerConfig) -> Dict[str, Any]:
    llm_input_v2 = build_llm_input_v2(cfg)
    out_path = Path(cfg.data_dir) / cfg.output_json
    out_path.write_text(json.dumps(llm_input_v2, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", out_path.resolve())
    return llm_input_v2


# =========================================================
# 4) Buffer-based loading (for Streamlit UploadedFile)
# =========================================================

def load_base_csv_from_buffer(buffer, cfg: AnalyzerConfig) -> pd.DataFrame:
    """Load base CSV from a file-like buffer (e.g. Streamlit UploadedFile)."""
    df = pd.read_csv(
        buffer,
        usecols=lambda c: c in USECOLS,
        dtype={k: v for k, v in SAFE_DTYPES.items() if k in USECOLS and k != "created"},
        parse_dates=["created"],
        encoding=cfg.csv_encoding,
        encoding_errors=cfg.csv_encoding_errors,
        sep=",",
        quotechar='"',
        doublequote=True,
        engine=cfg.csv_engine,
        on_bad_lines=cfg.csv_on_bad_lines,
    )

    required = {
        "created", "origin", "erp_invoice_id", "CT2_pack",
        "item_quantity", "revenues_wdisc_in_base_currency"
    }
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["item_quantity"] = pd.to_numeric(df["item_quantity"], errors="coerce")
    df["revenues_wdisc_in_base_currency"] = pd.to_numeric(df["revenues_wdisc_in_base_currency"], errors="coerce")

    if "net_margin_wdisc_in_base_currency" in df.columns:
        df["net_margin_wdisc_in_base_currency"] = pd.to_numeric(df["net_margin_wdisc_in_base_currency"], errors="coerce")
    else:
        df["net_margin_wdisc_in_base_currency"] = np.nan

    return df


def load_inventory_csv_from_buffer(buffer, cfg: AnalyzerConfig) -> Optional[pd.DataFrame]:
    """Load inventory CSV from a file-like buffer (e.g. Streamlit UploadedFile)."""
    if buffer is None:
        return None

    inv = pd.read_csv(
        buffer,
        usecols=lambda c: c in INVENTORY_USECOLS,
        encoding=cfg.inventory_encoding,
        engine=cfg.inventory_engine,
        on_bad_lines=cfg.inventory_on_bad_lines,
    )

    if "CT2_pack" not in inv.columns:
        raise ValueError("Inventory CSV must contain CT2_pack")

    inv["CT2_pack"] = inv["CT2_pack"].astype("string").fillna("")
    inv["stock_actual_quantity"] = pd.to_numeric(inv["stock_actual_quantity"], errors="coerce").fillna(0.0)

    agg = (
        inv.groupby("CT2_pack", dropna=False)
        .agg(on_hand_qty=("stock_actual_quantity", "sum"))
        .reset_index()
    )

    agg["on_hand_qty"] = agg["on_hand_qty"].round(0)
    agg["in_stock_flag"] = (agg["on_hand_qty"] > 0).astype(int)

    return agg


def build_llm_input_v2_from_buffers(main_buffer, inv_buffer, cfg: AnalyzerConfig) -> Dict[str, Any]:
    """Build the full analytics payload from file buffers instead of file paths."""
    df = load_base_csv_from_buffer(main_buffer, cfg)
    df2w = filter_two_weeks(df, cfg)
    df2w = add_period_flag(df2w, cfg)

    inv_pack = load_inventory_csv_from_buffer(inv_buffer, cfg) if inv_buffer else None
    df2w = attach_inventory(df2w, inv_pack)

    ct2_pack_weekly = compute_ct2_pack_weekly(df2w)
    presence = compute_presence(ct2_pack_weekly)

    revenue_bridge_summary = compute_revenue_bridge_summary(ct2_pack_weekly, presence)
    global_totals_check = compute_global_totals_check(df2w)
    common_pvm = compute_common_pvm(ct2_pack_weekly, presence)

    pack_quadrant = compute_pack_quadrant(ct2_pack_weekly)
    quads_summary = summarize_quadrants(pack_quadrant)
    quads_members = quadrant_members(pack_quadrant, top_n=80)

    basket_kpis_period = compute_basket_kpis_period(df2w)
    buyer_split_period = compute_buyer_split_period(df2w)

    volume_pack_contrib = compute_volume_contrib_top_packs_by_segment(df2w, top_n=3)
    pvm_row = common_pvm.iloc[0].to_dict()
    total_volume_effect = float(pvm_row.get("volume_effect", 0.0))
    volume_explanation = build_volume_explanation(
        basket_kpis_period,
        buyer_split_period,
        volume_pack_contrib,
        total_volume_effect=total_volume_effect,
    )

    price_pack_contrib = compute_price_contrib_top_packs_by_segment(df2w, top_n=3)
    price_explanation = build_price_explanation(buyer_split_period, price_pack_contrib)

    quadrant_impact_contrib = compute_quadrant_impact_contrib_top(df2w, top_n=20)
    quadrant_margin_contrib = compute_quadrant_margin_contrib_top(df2w, top_n=20)
    quadrant_key_products = compute_quadrant_key_products(df2w, top_n=6, coverage=0.7)

    global_kpis = compute_global_kpis(global_totals_check)
    math_check = compute_math_check(global_kpis, revenue_bridge_summary, cfg.bridge_tolerance)

    b = revenue_bridge_summary.iloc[0].to_dict()
    bridge_dict = {
        "appearing_total": float(b["appearing_total"]),
        "disappearing_total": float(b["disappearing_total"]),
        "common_delta_total": float(b["common_delta_total"]),
        "total_delta_from_bridge": float(b["total_delta_from_bridge"]),
    }

    pvm_row = common_pvm.iloc[0].to_dict()
    common_pvm_dict = {
        "delta_common_revenue": float(pvm_row.get("delta_common_revenue", 0.0)),
        "volume_effect": float(pvm_row.get("volume_effect", 0.0)),
        "price_effect": float(pvm_row.get("price_effect", 0.0)),
        "mix_effect": float(pvm_row.get("mix_effect", 0.0)),
    }

    inventory_kpis = {}
    if inv_pack is not None and len(inv_pack) > 0:
        inventory_kpis = {
            "packs_total": int(len(inv_pack)),
            "packs_in_stock": int((inv_pack["in_stock_flag"] == 1).sum()),
            "total_on_hand_qty": float(np.round(inv_pack["on_hand_qty"].sum(), 0)),
        }

    margin_kpis = compute_margin_kpis(df2w)

    llm_input_v2 = {
        "periods": {
            "t1": {"start": cfg.t1_start, "end": cfg.t1_end},
            "t2": {"start": cfg.t2_start, "end": cfg.t2_end},
        },
        "global_kpis": global_kpis,
        "bridge": bridge_dict,
        "math_check": math_check,
        "common_pvm": common_pvm_dict,
        "quadrants": {"summary": quads_summary, "members_ct2_pack": quads_members},
        "volume_explanation": volume_explanation,
        "price_explanation": price_explanation,
        "quadrant_impact_contrib": quadrant_impact_contrib,
        "quadrant_margin_contrib": quadrant_margin_contrib,
        "quadrant_key_products": quadrant_key_products,
        "margin_kpis": margin_kpis,
        "inventory_kpis": inventory_kpis,
        "inventory_meta": {
            "source": "uploaded",
            "note": "Inventory is a snapshot joined by CT2_pack (no history). Use only for action buckets.",
            "fields": ["on_hand_qty", "in_stock_flag"],
        },
        "_meta": {
            "source": "Uploaded CSV",
            "filters": {"origin": cfg.origin, "t1": [cfg.t1_start, cfg.t1_end], "t2": [cfg.t2_start, cfg.t2_end]},
            "notes": [
                "All computed in pandas from the raw export.",
                "PVM computed on COMMON packs only; appearing/disappearing handled in bridge.",
            ],
        },
    }

    return llm_input_v2


if __name__ == "__main__":
    cfg = AnalyzerConfig(
        data_dir=".",
        input_csv="abbb.csv",
        inventory_csv="BASE_TABLE_INVENTORY.csv",
        output_json="llm_input_v2.json",
        csv_encoding="cp1250",
        csv_engine="python",
        csv_on_bad_lines="skip",
    )
    run_analyzer(cfg)

