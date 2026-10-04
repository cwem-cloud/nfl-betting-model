"""Pick rationale: a deterministic summary of how the number was derived, optionally
polished into prose by Claude when ANTHROPIC_API_KEY is set."""
from __future__ import annotations

from pocket_capper.market.odds_math import fmt_american
from pocket_capper.settings import config, secret


def build(p: dict) -> str:
    """p: pick dict with blind/market/signal context. Returns markdown bullets."""
    lines = []
    cat = p["category"]
    if cat in ("side", "total", "ml"):
        lines.append(
            f"**Blind handicap (no line):** {p['blind_summary']}"
        )
        lines.append(
            f"**Market:** DK {p['dk_summary']} | sharp consensus {p.get('sharp_summary') or 'n/a'}"
        )
    else:
        lines.append(f"**Projection (no line):** {p['blind_summary']}")
        lines.append(f"**Market:** DK {p['dk_summary']}")
    lines.append(
        f"**Probabilities:** model {p['model_prob']:.1%}, market {p['market_prob']:.1%} "
        f"-> blended {p['final_prob']:.1%} vs {p['breakeven']:.1%} break-even at {fmt_american(p['price'])}"
        f" = **{p['ev']:+.1%} EV**"
    )
    sigs = p.get("signals") or []
    if sigs:
        agree = [f"{s['name']} ({s['detail']})" for s in sigs if s["direction"] > 0]
        against = [f"{s['name']} ({s['detail']})" for s in sigs if s["direction"] < 0]
        if agree:
            lines.append("**Sharp/market support:** " + "; ".join(agree))
        if against:
            lines.append("**Against us:** " + "; ".join(against))
    for n in p.get("context") or []:
        lines.append(f"**Context:** {n}")
    lines.append(f"**Sizing:** {p['units']:g}U (¼-Kelly on blended edge, signal-adjusted)")
    return "\n".join(f"- {l}" for l in lines)


def polish(play: str, bullets: str) -> str:
    """One short paragraph from Claude; returns '' on any failure (bullets still shown)."""
    if not secret("ANTHROPIC_API_KEY"):
        return ""
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=secret("ANTHROPIC_API_KEY"))
        resp = client.beta.messages.create(
            model=config()["run"]["claude_model"],
            max_tokens=2000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low"},
            system=(
                "You are a sharp sports-betting analyst. Write a 2-3 sentence rationale for the play "
                "using only the facts provided. No hype, no guarantees, no new facts."
            ),
            messages=[{"role": "user", "content": f"Play: {play}\nFacts:\n{bullets}"}],
        )
        if resp.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in resp.content if b.type == "text").strip()
    except Exception:
        return ""
