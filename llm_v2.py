# llm_v2.py
# NOTE: This is a self-contained version that includes:
# - make_pretty()
# - render_human_report()
# - Volume repeat/one_time lines rendered from *effects* (volume_effect_total + share_of_total_volume_effect_pct)
# so you will NOT get "revenue Δ=n/a" in the Volume section anymore.

import json
import re
from typing import Any, Dict, Optional, List
from pathlib import Path

from llama_cpp import Llama


# =========================================================
# 0) CONFIG
# =========================================================
MODEL_PATH = r"C:\Users\laci\Models\llama-3.1-8b-instruct-q4_k_m.gguf"
INPUT_JSON = "llm_input_v2.json"
OUTPUT_TXT = "weekly_summary_v2.txt"


# =========================================================
# 1) Robust JSON extraction / parsing
# =========================================================
def extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start: i + 1]
    return None


def json_loads_robust(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise json.JSONDecodeError("Empty response", text, 0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    extracted = extract_first_json_object(text)
    if extracted:
        return json.loads(extracted)

    raise json.JSONDecodeError("No JSON object found", text, 0)



def repair_json_with_llm(llm: Llama, raw_text: str) -> Dict[str, Any]:
    system_message = (
        "You fix invalid JSON.\n"
        "Return ONLY a single valid JSON object.\n"
        "Rules:\n"
        "- Output must start with '{' and end with '}'\n"
        "- Use double quotes for all keys/strings\n"
        "- No trailing commas\n"
        "- No markdown, no comments, no extra text\n"
        "- Validate JSON mentally before returning\n"
    )
    user_message = (
        "Fix this into VALID JSON. Keep the same fields if possible.\n\n"
        "RAW:\n" + (raw_text or "")
    )

    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        top_p=1.0,
        repeat_penalty=1.05,
        max_tokens=1400,   # <<< PATCH: nagyobb
    )
    fixed = (resp["choices"][0]["message"]["content"] or "").strip()
    _write_debug("weekly_llm_fixed.txt", fixed)  # <<< PATCH: debug
    return json_loads_robust(fixed)



# =========================================================
# 2) Helpers
# =========================================================
def _shorten_list(xs, n=8):
    xs = [str(x) for x in (xs or [])]
    if len(xs) <= n:
        return xs
    return xs[:n] + [f"... (+{len(xs)-n} more)"]


def normalize_executive_summary(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[🔴👉•]", "", text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    cleaned = []
    for l in lines:
        l = re.sub(r"^\d+[\.\)]\s*", "", l)
        cleaned.append(l)
    cleaned = cleaned[:4]
    return " ".join(cleaned)


def fmt_int(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(round(float(x), 0))
    except Exception:
        return None


def _fmt_money(x) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except Exception:
        return "n/a"
    s = f"{int(round(v)):,}".replace(",", " ")
    return f"{s} Ft"


def _fmt_pct(x) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.1f}%"
    except Exception:
        return "n/a"

def _fmt_table_row(cols: List[str], widths: List[int]) -> str:
    """Fixed-width table row with safe truncation."""
    out = []
    for c, w in zip(cols, widths):
        s = "" if c is None else str(c)
        s = s.replace("\n", " ").strip()
        if len(s) > w:
            s = s[: max(0, w - 3)] + "..."
        out.append(s.ljust(w))
    return " | ".join(out)


def top_quadrant_packs(llm_input: Dict[str, Any], quadrant: str, n: int = 30) -> List[str]:
    members = (llm_input.get("quadrants", {}) or {}).get("members_ct2_pack", {}) or {}
    packs = members.get(quadrant, []) or []
    packs = [str(x) for x in packs if x is not None]
    return packs[:n]


def _write_debug(path: str, text: str) -> None:
    try:
        Path(path).write_text(text or "", encoding="utf-8")
    except Exception:
        # ne álljon el a pipeline debug-írás miatt
        pass

def _suggest_action_from_pv_margin(price_effect: float, volume_effect: float, delta_margin: float) -> str:
    """
    Simple deterministic rule:
    - If inventory later comes in, we can enrich this. For now PV + margin only.
    """
    pe = price_effect or 0.0
    ve = volume_effect or 0.0
    dm = delta_margin or 0.0

    # Price↑ Volume↓ quadrant typical:
    if pe > 0 and ve < 0:
        if dm < 0:
            return "PRICE DOWN TEST / PROMO"
        if dm > 0:
            return "KEEP / PRICE-UP (controlled)"
        return "MONITOR (borderline)"

    # Price↓ Volume↑
    if pe < 0 and ve > 0:
        if dm > 0:
            return "PROMO SCALE"
        if dm < 0:
            return "PROMO STOP / PRICE UP TEST"
        return "MONITOR"

    # Double hit
    if pe < 0 and ve < 0:
        if dm < 0:
            return "DELIST / FIX AVAILABILITY"
        return "PRICE REVIEW"

    # Growth zone
    if pe > 0 and ve > 0:
        if dm > 0:
            return "KEEP / PRICE-UP"
        return "KEEP (watch margin)"

    return "MONITOR"


def select_items_by_coverage(
    items: List[dict],
    coverage_target_pct: float = 60.0,
    min_items: int = 4,
    max_items: int = 12,
) -> Dict[str, Any]:
    """
    Coverage-based selection using abs(impact_effect).
    Returns dict with:
      - selected: List[dict]
      - achieved_coverage_pct: float
      - hit_cap: bool (True if max_items reached before hitting target)
      - universe_abs: float
      - selected_abs: float
    """
    if not items:
        return {
            "selected": [],
            "achieved_coverage_pct": 0.0,
            "hit_cap": False,
            "universe_abs": 0.0,
            "selected_abs": 0.0,
        }

    # Always sort here (do NOT assume upstream sorting)
    items_sorted = sorted(
        items,
        key=lambda it: abs(float(it.get("impact_effect") or 0.0)),
        reverse=True
    )

    universe_abs = sum(abs(float(it.get("impact_effect") or 0.0)) for it in items_sorted)

    # If universe is zero, just return up to min_items (no meaningful coverage)
    if universe_abs <= 0:
        sel = items_sorted[: min(min_items, len(items_sorted))]
        return {
            "selected": sel,
            "achieved_coverage_pct": 0.0,
            "hit_cap": False,
            "universe_abs": universe_abs,
            "selected_abs": 0.0,
        }

    selected: List[dict] = []
    cum_abs = 0.0
    hit_cap = False

    for it in items_sorted:
        if len(selected) >= max_items:
            hit_cap = True
            break

        selected.append(it)
        cum_abs += abs(float(it.get("impact_effect") or 0.0))

        achieved = (cum_abs / universe_abs) * 100.0
        if len(selected) >= min_items and achieved >= coverage_target_pct:
            break

    achieved = (cum_abs / universe_abs) * 100.0 if universe_abs > 0 else 0.0

    return {
        "selected": selected,
        "achieved_coverage_pct": achieved,
        "hit_cap": hit_cap,
        "universe_abs": universe_abs,
        "selected_abs": cum_abs,
    }



# =========================================================
# 3) Payload builder
# =========================================================
def build_llm_payload_v2(llm_input: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "periods": llm_input.get("periods"),
        "global_kpis": llm_input.get("global_kpis"),
        "bridge": llm_input.get("bridge"),
        "common_pvm": llm_input.get("common_pvm"),
        "quadrants_summary": (llm_input.get("quadrants", {}) or {}).get("summary", []),

        "volume_explanation": llm_input.get("volume_explanation", {}),
        "price_explanation": llm_input.get("price_explanation", {}),

        "margin_kpis": llm_input.get("margin_kpis", {}),
        "inventory_kpis": llm_input.get("inventory_kpis", {}),
        "inventory_meta": llm_input.get("inventory_meta", {}),

        "example_packs": {
            "D4_price_down_volume_down": top_quadrant_packs(llm_input, "D4: Price↓ Volume↓", n=8),
            "D2_price_up_volume_down": top_quadrant_packs(llm_input, "D2: Price↑ Volume↓", n=8),
        },
    }


# =========================================================
# 4) Main LLM call (story + buckets)
# =========================================================
def llama_story_v2(llm: Llama, llm_input: Dict[str, Any]) -> Dict[str, Any]:
    payload = build_llm_payload_v2(llm_input)

    system_message = (
        "Te egy senior performance analyst vagy. "
        "Feladat: heti ok-okozati összefoglalót írni a bemenet alapján.\n"
        "SZIGORÚ SZABÁLYOK:\n"
        "1) Csak a bemenetben szereplő számokra támaszkodhatsz.\n"
        "2) Nem találhatsz ki új metrikát.\n"
        "3) TILOS audit/QA témát említeni (se story, se findings, se actions).\n"
        "4) A kimenet KIZÁRÓLAG érvényes JSON legyen, semmi extra szöveg.\n"
        "5) A VOLUME fejezetben TILOS 'revenue Δ' / 'delta revenue' / 'n/a' kifejezés.\n"
        "6) Az executive_summary NEM lehet üres: 6–9 rövid bullet sor, '- ' kezdettel.\n"
        "   Témák: mi történt (ΔR), mi dominált (Bridge+PVM), hol (Quadrants), és mi a fókusz (Actions).\n"
        "   Ott kizárólag Volume hatás és annak total-on belüli share-e szerepelhet.\n"
        "Return ONLY valid JSON. No markdown. No explanation. No extra keys. No extra text.\n"
    )

    user_message = (
        "A kimenet JSON objektum PONTOSAN ezekkel a mezőkkel:\n"
        "{\n"
        '  "headline": string,\n'
        '  "executive_summary": [ string ],\n'
        '  "story": string,\n'
        '  "top_findings": [ string ],\n'
        '  "recommended_actions": [ string ],\n'
        '  "action_buckets": {\n'
        '    "KEEP / PRICE-UP": [string],\n'
        '    "PROMO SCALE / PROMO STOP": [string],\n'
        '    "PRICE DOWN TEST": [string],\n'
        '    "FIX AVAILABILITY": [string],\n'
        '    "DELIST": [string]\n'
        '  },\n'
        '  "diagnostic_packs": { "D4": [string], "D2": [string] }\n'
        "}\n\n"

        "EXECUTIVE_SUMMARY SZABÁLY:\n"
        "- 'executive_summary' egy 6–9 elemű lista legyen.\n"
        "- Minden elem 1 sor, rövid, üzleti tanulság jellegű.\n"
        "- Csak a bemeneti számokra hivatkozhatsz.\n\n"

        "VOLUME RÉSZ:\n"
        "- Repeat vs one_time bontásnál KIZÁRÓLAG ezek a mezők használhatók szegmensenként:\n"
        "  volume_explanation.buyer_split.<seg>.volume_effect_total\n"
        "  volume_explanation.buyer_split.<seg>.share_of_total_volume_effect_pct\n"
        "- TILOS bármilyen revenue / delta / n/a szöveg.\n"
        "- Kötelező formátum szegmensenként:\n"
        "  repeat_buyer=<seg>: orders t1=... → t2=... (Δ=...), Volume hatás = ... Ft (...% a teljes Volume hatásból)\n\n"

        "5 BUCKET SZABÁLYOK:\n"
        "- Mind az 5 bucket legyen jelen (ha nincs adat: []).\n"
        "- Csak a bemenet alapján javasolj.\n"
        "- Használd a quadrants_summary és a diagnostic_packs listákat példáknak.\n"
        "- FIX AVAILABILITY bucketbe csak akkor írj, ha inventory_kpis alapján van készlet-probléma jel.\n"
        "- Margin_kpis-t használd: ha margin romlik → PRICE DOWN TEST / PROMO felé; ha stabil/javul → óvatosan KEEP/PRICE-UP.\n\n"

        "BEMENET:\n"
        + json.dumps(payload, ensure_ascii=False)
    )

    def _call_story(max_tokens: int) -> str:
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            top_p=1.0,
            repeat_penalty=1.10,
            max_tokens=max_tokens,
        )
        return (resp["choices"][0]["message"]["content"] or "").strip()

    raw = _call_story(max_tokens=1400)  # <<< PATCH: nagyobb
    # DEBUG: csak fájlba mentünk, konzolra nem printelünk
    # print("\n=== RAW MODEL OUTPUT (first 2000 chars) ===\n")
    # print(raw[:2000])
    # print("\n=== RAW MODEL OUTPUT (last 500 chars) ===\n")
    # print(raw[-500:])

    _write_debug("weekly_llm_raw.txt", raw)  # <<< PATCH: debug

    # 1) Parse attempt
    try:
        out = json_loads_robust(raw)
    except json.JSONDecodeError:
        # 2) Repair attempt
        try:
            out = repair_json_with_llm(llm, raw)
        except json.JSONDecodeError:
            # 3) One retry with larger output budget (often fixes truncation)
            raw2 = _call_story(max_tokens=2000)
            _write_debug("weekly_llm_raw_retry.txt", raw2)
            try:
                out = json_loads_robust(raw2)
            except json.JSONDecodeError:
                # last try: repair the retry output
                out = repair_json_with_llm(llm, raw2)

    # deterministic: always attach diagnostic packs
    out["diagnostic_packs"] = {
        "D4": payload["example_packs"]["D4_price_down_volume_down"],
        "D2": payload["example_packs"]["D2_price_up_volume_down"],
    }

    # ensure buckets exist
    if "action_buckets" not in out or not isinstance(out["action_buckets"], dict):
        out["action_buckets"] = {
            "KEEP / PRICE-UP": [],
            "PROMO SCALE / PROMO STOP": [],
            "PRICE DOWN TEST": [],
            "FIX AVAILABILITY": [],
            "DELIST": [],
        }
    else:
        for k in ["KEEP / PRICE-UP", "PROMO SCALE / PROMO STOP", "PRICE DOWN TEST", "FIX AVAILABILITY", "DELIST"]:
            out["action_buckets"].setdefault(k, [])

    out["_debug_payload"] = payload
    out["_debug_raw_model_output"] = raw if len(raw) < 3500 else raw[:3500] + " ...[truncated]"
    return out


# =========================================================
# 5) Exec summary from deterministic report (optional)
# =========================================================
def llama_exec_summary_from_report(llm: Llama, data_block_text: str) -> str:
    system_message = (
        "Te egy senior FP&A / controlling vezető vagy. "
        "Feladat: tömör, döntéstámogató Executive Summary a DATA BLOCK alapján.\n\n"
        "KÖTELEZŐ SZABÁLYOK:\n"
        "1) Ne mondd fel a riportot. Ne ismételd magad.\n"
        "2) Ugyanazt a számot vagy tényt MAXIMUM 1× említheted az egész szövegben.\n"
        "   (Példa: ha leírtad a repeat 86,2%-ot, máshol nem írhatod le újra.)\n"
        "3) Pontosan 4 bekezdés legyen, pontosan ezekkel a címsorokkal:\n"
        "   A) Mi történt?\n"
        "   B) Mi okozta?\n"
        "   C) Hol történt?\n"
        "   D) Mit javaslunk?\n"
        "4) Pontosan 6 mondat összesen az egész válaszban.\n"
        "   - A) 2 mondat\n"
        "   - B) 2 mondat\n"
        "   - C) 1 mondat\n"
        "   - D) 1 mondat\n"
        "5) Minden bekezdésben legyen legalább 1 szám (Ft vagy %).\n"
        "6) Terméknevek/CT2_pack-ek: ALAPÉRTELMEZÉS: 0 db. "
        "MAX 1 terméknév említhető összesen, csak ha több szempontból is bizonyíthatóan jelentős a DATA BLOCK alapján.\n"
        "7) Ha a termékmix releváns a DATA BLOCK alapján, használd ezt a kifejezést egyszer:\n"
        "   \"termékmix-hatás (újonnan megjelenő és eltűnő termékek egyenlege)\"\n"
        "8) Formátum: ne bullet-lista, hanem mondatok.\n"
        "9) D) 'Mit javaslunk?' rész: KIZÁRÓLAG a Quadrants (D1–D4) tanulságai alapján adhatsz javaslatot.\n"
        "   - TILOS bármilyen terméknév / CT2_pack / key decision products hivatkozás D-ben.\n"
        "   - TILOS Volume/Price magyarázat (basket/repeat) alapján érvelni D-ben.\n"
        "   - Csak ilyen jellegű, kvadráns-logika megengedett:\n"
        "     D4: Price↓ Volume↓ → double hit: FIX AVAILABILITY / DELIST / alap probléma kezelése\n"
        "     D2: Price↑ Volume↓ → price-down test / kontrollált korrekció\n"
        "     D3: Price↓ Volume↑ → promo scale, de margin kontroll\n"
        "     D1: Price↑ Volume↑ → skálázás / ismétlés\n"
        "10) D) részben kötelező legalább 1 kvadráns kód megnevezése (D1/D2/D3/D4).\n"
        "11) PÉNZEGYSÉG SZABÁLY: 'milliárd' szót TILOS használni 1 000 000 000 Ft alatt.\n"
        "   Ha nagyságrend < 1 000 000 000 Ft, akkor 'millió Ft' / 'M Ft' formátumot használj.\n"



    )

    user_message = (
        "Írj Executive Summary-t az alábbi DATA BLOCK alapján.\n\n"
        "DATA BLOCK (immutable)\n"
        + (data_block_text or "")
        + "\nEND DATA BLOCK\n"
    )

    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        top_p=1.0,
        # Anti-loop: ezek segítenek, ha a modell hajlamos ismételni
        frequency_penalty=0.35,
        presence_penalty=0.10,
        repeat_penalty=1.10,
        max_tokens=460,  # 6 mondat + 4 cím simán belefér
        stop=["<END>"]
    )
    return (resp["choices"][0]["message"]["content"] or "").strip()


# =========================================================
# 6) Pretty report (deterministic)
# =========================================================
def make_pretty(llm_input: Dict[str, Any], weekly: dict) -> dict:
    g = llm_input.get("global_kpis", {}) or {}
    bridge = llm_input.get("bridge", {}) or {}
    pvm = llm_input.get("common_pvm", {}) or {}
    volx = llm_input.get("volume_explanation", {}) or {}

    return {
        "headline": weekly.get("headline", "Heti összefoglaló"),
        "periods": llm_input.get("periods", {}),
        "executive_summary": weekly.get("executive_summary", ""),
        "story": weekly.get("story", ""),
        "global": {
            "t1_revenue": fmt_int(g.get("t1_revenue")),
            "t2_revenue": fmt_int(g.get("t2_revenue")),
            "delta_revenue": fmt_int(g.get("delta_revenue")),
        },
        "revenue_bridge": {
            "appearing_t2_only": fmt_int(bridge.get("appearing_total")),
            "disappearing_t1_only": fmt_int(bridge.get("disappearing_total")),
            "common_delta": fmt_int(bridge.get("common_delta_total")),
            "total_delta": fmt_int(bridge.get("total_delta_from_bridge")),
        },
        "common_pvm": {
            "delta_common_revenue": fmt_int(pvm.get("delta_common_revenue")),
            "volume": fmt_int(pvm.get("volume_effect")),
            "price": fmt_int(pvm.get("price_effect")),
            "mix": fmt_int(pvm.get("mix_effect")),
        },
        "volume_explanation": volx,
        "price_explanation": llm_input.get("price_explanation", {}),
        "quadrants_summary": (llm_input.get("quadrants", {}) or {}).get("summary", []),
        "quadrant_impact_contrib": llm_input.get("quadrant_impact_contrib", {}),
        "quadrant_margin_contrib": llm_input.get("quadrant_margin_contrib", {}),
        "quadrant_key_products": llm_input.get("quadrant_key_products", {}),

        "top_findings": (weekly.get("top_findings") or [])[:5],
        "recommended_actions": (weekly.get("recommended_actions") or [])[:5],
        "diagnostic_packs_short": {
            "D4": _shorten_list((weekly.get("diagnostic_packs") or {}).get("D4", []), n=8),
            "D2": _shorten_list((weekly.get("diagnostic_packs") or {}).get("D2", []), n=8),
        },
        "diagnostic_products_by_quadrant": {
            "D1: Price↑ Volume↑": _shorten_list(((llm_input.get("quadrants", {}) or {}).get("members_ct2_pack", {}) or {}).get("D1: Price↑ Volume↑", []), n=8),
            "D2: Price↑ Volume↓": _shorten_list(((llm_input.get("quadrants", {}) or {}).get("members_ct2_pack", {}) or {}).get("D2: Price↑ Volume↓", []), n=8),
            "D3: Price↓ Volume↑": _shorten_list(((llm_input.get("quadrants", {}) or {}).get("members_ct2_pack", {}) or {}).get("D3: Price↓ Volume↑", []), n=8),
            "D4: Price↓ Volume↓": _shorten_list(((llm_input.get("quadrants", {}) or {}).get("members_ct2_pack", {}) or {}).get("D4: Price↓ Volume↓", []), n=8),
        },
    }


def _render_volume_buyer_split_lines(volx: dict) -> List[str]:
    """
    Volume buyer split lines + Top3 packs PER SEGMENT directly under the segment line.
    Uses:
      - volx["buyer_split"][seg]["volume_effect_total"]
      - volx["buyer_split"][seg]["share_of_total_volume_effect_pct"]
      - volx["volume_pack_contrib"][seg]["top_packs"]  (share is within-segment already)
    """
    buyer = (volx or {}).get("buyer_split", {}) or {}
    vcontrib = (volx or {}).get("volume_pack_contrib", {}) or {}

    lines: List[str] = []
    lines.append(" - Repeat vs one-time bontás:")

    for seg in ["one_time", "repeat"]:
        if seg not in buyer:
            continue

        s = buyer.get(seg, {}) or {}
        t1o = s.get("t1_orders")
        t2o = s.get("t2_orders")
        do = s.get("delta_orders")

        vol_total = s.get("volume_effect_total")
        share_total = s.get("share_of_total_volume_effect_pct")

        # 1) segment header line
        lines.append(
            f"   - repeat_buyer={seg}: orders t1={t1o} → t2={t2o} (Δ={do}), "
            f"Volume-hatás = {_fmt_money(vol_total)} ({_fmt_pct(share_total)} a teljes Volume-hatásból)"
        )

        # 2) Top packs right under this segment
        seg_obj = (vcontrib.get(seg) or {})
        top_neg = (seg_obj.get("top_packs_neg") or [])
        top_pos = (seg_obj.get("top_packs_pos") or [])

        if top_neg:
            lines.append("     Top3 negatív CT2_pack a Volume-hatásból (worst offenders, szegmensen belül):")
            for i, it in enumerate(top_neg[:3], start=1):
                lines.append(
                    f"       {i}) {it.get('CT2_pack')}: "
                    f"{_fmt_money(it.get('volume_effect'))} ({_fmt_pct(it.get('share'))})"
                )

        if top_pos:
            lines.append("     Top3 pozitív CT2_pack a Volume-hatásból (best contributors, szegmensen belül):")
            for i, it in enumerate(top_pos[:3], start=1):
                lines.append(
                    f"       {i}) {it.get('CT2_pack')}: "
                    f"{_fmt_money(it.get('volume_effect'))} ({_fmt_pct(it.get('share'))})"
                )

    return lines


def render_human_report(pretty: dict, include_exec_summary: bool = True) -> str:
    lines: List[str] = []
    lines.append(pretty.get("headline", "Heti összefoglaló"))
    lines.append("")

    # -----------------------------
    # Period header (t1 / t2)
    # -----------------------------
    periods = pretty.get("periods", {}) or {}
    t1 = (periods.get("t1", {}) or {})
    t2 = (periods.get("t2", {}) or {})
    
    t1s, t1e = t1.get("start"), t1.get("end")
    t2s, t2e = t2.get("start"), t2.get("end")
    
    if t1s and t1e and t2s and t2e:
        lines.append(f"Időszakok: t1 = {t1s} – {t1e} | t2 = {t2s} – {t2e}")
        lines.append("")

    
    # -----------------------------
    # Global summary
    # -----------------------------
    g = pretty.get("global", {}) or {}
    lines.append("Összefoglaló")
    lines.append(f" - Összes árbevétel: t1 = {_fmt_money(g.get('t1_revenue'))}")
    lines.append(f"                   t2 = {_fmt_money(g.get('t2_revenue'))}")
    lines.append(f"                   → Δ = {_fmt_money(g.get('delta_revenue'))}.")
    lines.append("")

    # -----------------------------
    # Executive summary (LLM)
    # -----------------------------
    if include_exec_summary:
        es = (pretty.get("executive_summary") or "").strip()
        if es:
            lines.append("Executive summary (LLM értelmezés)")
            for ln in es.splitlines():
                ln = ln.rstrip()
                if ln:
                    lines.append(f" {ln}")
            lines.append("")


    # -----------------------------
    # Revenue bridge
    # -----------------------------
    b = pretty.get("revenue_bridge", {}) or {}
    lines.append("Revenue bridge")
    lines.append(f" - Megjelenő (t2-only): {_fmt_money(b.get('appearing_t2_only'))}")
    lines.append(f" - Eltűnő (t1-only): {_fmt_money(b.get('disappearing_t1_only'))}")
    lines.append(f" - Közös termékek Δ: {_fmt_money(b.get('common_delta'))}")
    lines.append(f" - Összes Δ: {_fmt_money(b.get('total_delta'))}")
    lines.append("")

    # -----------------------------
    # PVM
    # -----------------------------
    pvm = pretty.get("common_pvm", {}) or {}
    lines.append("PVM (Price–Volume–Mix) – közös termékek")
    lines.append(f" - ΔR (common revenue): {_fmt_money(pvm.get('delta_common_revenue'))}  [ΔR = Σ(r2 − r1)]")
    lines.append(f" - Volume-hatás:        {_fmt_money(pvm.get('volume'))}  [(q2 − q1) × p1, ahol p1 = r1/q1]")
    lines.append(f" - Price-hatás:         {_fmt_money(pvm.get('price'))}  [q2 × (p2 − p1), ahol p2 = r2/q2]")
    lines.append(f" - Mix-hatás:           {_fmt_money(pvm.get('mix'))}  [ΔR − Volume − Price]")
    lines.append("")

    # -----------------------------
    # Volume explanation
    # -----------------------------
    volx = pretty.get("volume_explanation", {}) or {}
    lines.append("Volume-hatás magyarázat (kosár + repeat bontás + top pack hozzájárulók)")

    basket = (volx.get("basket") or {})
    if basket:
        lines.append(
            f" - Order count (invoice_cnt): "
            f"t1={basket.get('t1', {}).get('invoice_cnt')}, "
            f"t2={basket.get('t2', {}).get('invoice_cnt')}, "
            f"Δ={basket.get('delta', {}).get('invoice_cnt')}"
        )
        lines.append(
            f" - Kosárméret (units/invoice): "
            f"t1={basket.get('t1', {}).get('units_per_invoice')}, "
            f"t2={basket.get('t2', {}).get('units_per_invoice')}, "
            f"Δ={basket.get('delta', {}).get('units_per_invoice')}"
        )
        lines.append(" - Következtetés: order-driven (nem a kosár csökkent, hanem a rendelésszám).")

    # >>> FIX HERE: buyer split rendered from Volume EFFECTS, not revenue delta
    lines.extend(_render_volume_buyer_split_lines(volx))

    lines.append("")

    # -----------------------------
    # Price explanation
    # -----------------------------
    px = pretty.get("price_explanation", {}) or {}
    pcontrib = (px.get("price_pack_contrib") or {})  # by_segment

    if pcontrib:
        lines.append("Price-hatás magyarázat (repeat bontás + top pack hozzájárulók)")

        total_price = (pretty.get("common_pvm", {}) or {}).get("price")
        denom = abs(total_price) if (total_price is not None and abs(total_price) > 0) else None

        for seg in ["one_time", "repeat"]:
            seg_obj = (pcontrib.get(seg) or {})
            seg_total = seg_obj.get("segment_price_effect_total")

            if seg_total is None and not seg_obj:
                continue

            if seg_total is not None:
                share_total = None
                if denom:
                    share_total = (abs(float(seg_total)) / float(denom)) * 100.0

                lines.append(
                    f" - repeat_buyer={seg}: Price-hatás = {_fmt_money(seg_total)} "
                    f"({_fmt_pct(share_total)} a teljes Price-hatásból)"
                )

            top_neg = (seg_obj.get("top_packs_neg") or [])
            top_pos = (seg_obj.get("top_packs_pos") or [])

            if top_neg:
                lines.append("     Top3 negatív CT2_pack a Price-hatásból (worst offenders, szegmensen belül):")
                for i, it in enumerate(top_neg[:3], start=1):
                    lines.append(
                        f"       {i}) {it.get('CT2_pack')}: "
                        f"{_fmt_money(it.get('price_effect'))} ({_fmt_pct(it.get('share'))})"
                    )

            if top_pos:
                lines.append("     Top3 pozitív CT2_pack a Price-hatásból (best contributors, szegmensen belül):")
                for i, it in enumerate(top_pos[:3], start=1):
                    lines.append(
                        f"       {i}) {it.get('CT2_pack')}: "
                        f"{_fmt_money(it.get('price_effect'))} ({_fmt_pct(it.get('share'))})"
                    )

        lines.append("")

    
    # -----------------------------
    # Price × Volume dynamics (Quadrants)
    # -----------------------------
    quads = pretty.get("quadrants_summary", []) or []
    qcontrib = pretty.get("quadrant_impact_contrib", {}) or {}
    qmarg = pretty.get("quadrant_margin_contrib", {}) or {}
    qkeys = pretty.get("quadrant_key_products", {}) or {}

    if quads:
        lines.append("Ár–mennyiség dinamika + diagnosztikai példák (Price × Volume együtt)")

        quad_name = {
            "D1: Price↑ Volume↑": "„Növekedési zóna” (Price↑, Volume↑)",
            "D2: Price↑ Volume↓": "„Árrés-javító, de volumen-kockázatos” (Price↑, Volume↓)",
            "D3: Price↓ Volume↑": "„Promóciós növekedés” (Price↓, Volume↑)",
            "D4: Price↓ Volume↓": "„Kettős visszaesés (double hit)” (Price↓, Volume↓)",
            "Neutral / No change": "„Semleges / nincs érdemi változás”",
        }

        for row in quads:
            quad = row.get("quadrant")
            title = quad_name.get(quad, str(quad))

            pack_cnt = row.get("pack_cnt")
            delta_qty = row.get("delta_qty")

            lines.append(f" - {title}")
            lines.append(f"   - Érintett CT2_pack-ek: {pack_cnt}")
            lines.append(f"   - Volumen változás (összes): Δqty = {int(delta_qty) if delta_qty is not None else 'n/a'}")

            qc = qcontrib.get(str(quad), {}) or {}
            impact_total = qc.get("impact_total")
            mix_total = qc.get("mix_total")
            delta_rev_total = qc.get("delta_rev_total")

            lines.append(f"   - Impact (Price+Volume): {_fmt_money(impact_total)}")
            # MIX-et kvadránsonként NEM mutatjuk (pack-konzisztens kvadráns)
            lines.append(f"   - Árbevétel változás:    ΔR = {_fmt_money(delta_rev_total)}")


            # Top contrib (impact alapú) – ha van
            top = (qc.get("top_products") or [])[:3]
            if top:
                top1 = top[0]
                lines.append(
                    f"   - Legnagyobb impact Ct2_pack: "
                    f"{top1.get('CT2_pack')} ({_fmt_money(top1.get('impact_effect'))}, {_fmt_pct(top1.get('share_pct'))})"
                )

            # Margin quadrant (improver/detractor)
            mc = qmarg.get(str(quad), {}) or {}
            quad_delta_margin = mc.get("quadrant_delta_margin_total")

            if quad_delta_margin is not None:
                lines.append(f"   - Margin Δ (nettó):        {_fmt_money(quad_delta_margin)}")

                imp = mc.get("top_improver")
                det = mc.get("top_detractor")

                if imp:
                    lines.append(
                        f"   - Top margin improver:    {imp.get('CT2_pack')} ({_fmt_money(imp.get('delta_margin'))})"
                    )
                if det:
                    lines.append(
                        f"   - Top margin detractor:   {det.get('CT2_pack')} ({_fmt_money(det.get('delta_margin'))})"
                    )

            # -----------------------------
            # Key decision products (PV + Margin together)  [SINGLE-LINE TABLE]
            # -----------------------------
            key = (qkeys.get(str(quad), {}) or {})
            items = (key.get("items") or [])
            cov = key.get("coverage_pct")
            
            if items:
                cov_txt = f"{cov}%" if cov is not None else "n/a"
                lines.append(f"   - Key decision products (Top impact, coverage ~{cov_txt}):")
            
                # widths tuned for Notepad + Jupyter
                w_name = 46
                w_num = 12
                w_action = 24
            
                # header
                header = _fmt_table_row(
                    ["CT2_pack", "Price", "Volume", "Impact", "Margin Δ", "Action"],
                    [w_name, w_num, w_num, w_num, w_num, w_action]
                )
                sep = _fmt_table_row(
                    ["-" * w_name, "-" * w_num, "-" * w_num, "-" * w_num, "-" * w_num, "-" * w_action],
                    [w_name, w_num, w_num, w_num, w_num, w_action]
                )
                lines.append(f"     {header}")
                lines.append(f"     {sep}")
            
                for it in items[:6]:
                    pe = it.get("price_effect")
                    ve = it.get("volume_effect")
                    imp_eff = it.get("impact_effect")
                    dm = it.get("delta_margin")
            
                    act = _suggest_action_from_pv_margin(pe, ve, dm)
            
                    row = _fmt_table_row(
                        [
                            it.get("CT2_pack"),
                            _fmt_money(pe),
                            _fmt_money(ve),
                            _fmt_money(imp_eff),
                            _fmt_money(dm),
                            act,
                        ],
                        [w_name, w_num, w_num, w_num, w_num, w_action]
                    )
                    lines.append(f"     {row}")
            lines.append("")

        
        # -------------------------------------------------
        # Residual / Mix korrekció (egyetlen sorban)
        # -------------------------------------------------
        rec = (qcontrib.get("_reconciliation") or {})
        residual_mix = rec.get("residual_mix_total")

        # opcionális diagnosztika (ha megvan)
        unpriced_cnt = rec.get("unpriced_common_pack_cnt")
        unpriced_dR = rec.get("unpriced_common_delta_rev_total")

        if residual_mix is not None:
            # Ez legyen ugyanaz, mint common_pvm.mix (pl. -338 617 Ft)
            tail = ""
            if (unpriced_cnt is not None) and (unpriced_dR is not None):
                tail = f"  (unpriced common: {int(unpriced_cnt)} pack, ΔR={_fmt_money(unpriced_dR)})"

            lines.append(
                f" - Residual / Mix korrekció (common ΔR − Σ(Price+Volume), nem kvadráns-szinten bontva): "
                f"{_fmt_money(residual_mix)}{tail}"
            )
            lines.append("")

    
    # -----------------------------
    # Margin KPIs
    # -----------------------------
    mk = pretty.get("margin_kpis", {}) or {}
    if mk:
        lines.append("Margin (nettó)")
        lines.append(f" - Nettó margin: t1 = {_fmt_money(mk.get('t1_margin'))}")
        lines.append(f"               t2 = {_fmt_money(mk.get('t2_margin'))}")
        lines.append(f"               → Δ = {_fmt_money(mk.get('delta_margin'))}.")
        lines.append(f" - Margin% (nettó): t1 = {mk.get('t1_margin_pct')} | t2 = {mk.get('t2_margin_pct')}")
        lines.append("")

    return "\n".join(lines)

# =========================================================
# 7) Runner
# =========================================================
def main():
    inp_path = Path(INPUT_JSON)
    if not inp_path.exists():
        raise FileNotFoundError(f"Missing input JSON: {inp_path.resolve()}")

    llm_input = json.loads(inp_path.read_text(encoding="utf-8"))

    llm = Llama(
        model_path=MODEL_PATH,
        verbose=False,
        n_ctx=8192,
        n_threads=8,
        n_gpu_layers=35,
    )

    weekly = llama_story_v2(llm, llm_input)

    # 0) Pretty objektum létrehozása
    pretty = make_pretty(llm_input, weekly)

    # 1) Base report EXEC summary nélkül
    base_txt = render_human_report(pretty, include_exec_summary=False)

    # 2) LLM executive summary a kész riportból
    exec_sum = llama_exec_summary_from_report(llm, base_txt)
    pretty["executive_summary"] = exec_sum  # hagyjuk meg többsorosnak


    # 3) Final report EXEC summary-vel
    txt = render_human_report(pretty, include_exec_summary=True)
    print("DEBUG txt type:", type(txt))
    print("DEBUG txt head:", (txt or "")[:200])

    Path(OUTPUT_TXT).write_text(txt, encoding="utf-8")

    print("Saved:", Path(OUTPUT_TXT).resolve())
    print(txt)


if __name__ == "__main__":
    main()

