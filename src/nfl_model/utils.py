import numpy as np
import pandas as pd

def moneyline_to_prob(ml):
    # Convert American odds to implied probability (no vig removed)
    if ml < 0:
        return (-ml) / ((-ml) + 100.0)
    else:
        return 100.0 / (ml + 100.0)

def edge_to_confidence(edge, floor=0.0, ceiling=7.0):
    # Map edge (points or prob) to 1-10; tune this mapping later
    # Simple logistic-ish mapping
    val = 1 + 9 * (1 - np.exp(-abs(edge) / ceiling))
    return float(np.clip(val, 1, 10))

def pick_text(row):
    parts = []
    if row["bet_type"] == "ATS":
        side = "HOME" if row["pick_home"] else "AWAY"
        parts.append(f"{side} against the spread by {row['model_spread']:.1f} vs market {row['market_spread']:.1f}.")
    elif row["bet_type"] == "ML":
        side = "HOME" if row["pick_home"] else "AWAY"
        parts.append(f"{side} moneyline; model p={row['model_prob']:.2f} vs market p={row['market_prob']:.2f}.")
    else:
        side = "OVER" if row["pick_over"] else "UNDER"
        parts.append(f"{side} {row['model_total']:.1f} vs market {row['market_total']:.1f}.")
    parts.append("Key signals: QB adj, rolling EPA, rest/travel, and weather.")
    return " ".join(parts)
