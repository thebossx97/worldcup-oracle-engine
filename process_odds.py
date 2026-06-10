"""Entvigt die Markt-Quoten → faire Markt-Wahrscheinlichkeiten. Speichert market.json fürs Frontend.
Markt = stärkster Einzel-Prädiktor; unser Modell wird dagegen kalibriert (Ensemble/Shrinkage)."""
import json

def devig(prices):
    inv = [1.0 / p for p in prices]
    s = sum(inv)
    return [x / s for x in inv]

# ── Titel-Quoten (outrights) → Markt-Titelchancen ──
w = json.load(open("data/odds_winner.json"))
title = {}
if w and w[0].get("bookmakers"):
    # über mehrere Buchmacher mitteln (robuster)
    agg = {}
    for bk in w[0]["bookmakers"]:
        for mkt in bk["markets"]:
            if mkt["key"] != "outrights":
                continue
            names = [o["name"] for o in mkt["outcomes"]]
            probs = devig([o["price"] for o in mkt["outcomes"]])
            for n, p in zip(names, probs):
                agg.setdefault(n, []).append(p)
    title = {n: sum(ps) / len(ps) for n, ps in agg.items()}

# ── Spiel-Quoten (h2h) → Markt-W/U/N je Spiel ──
h = json.load(open("data/odds_h2h.json"))
matches = []
for ev in h:
    home, away = ev["home_team"], ev["away_team"]
    rows = []
    for bk in ev.get("bookmakers", []):
        for mkt in bk["markets"]:
            if mkt["key"] != "h2h":
                continue
            o = {x["name"]: x["price"] for x in mkt["outcomes"]}
            if home in o and away in o and "Draw" in o:
                ph, pd, pa = devig([o[home], o["Draw"], o[away]])
                rows.append((ph, pd, pa))
    if rows:
        n = len(rows)
        matches.append({
            "home": home, "away": away, "date": ev.get("commence_time", "")[:10],
            "ph": sum(r[0] for r in rows) / n, "pd": sum(r[1] for r in rows) / n,
            "pa": sum(r[2] for r in rows) / n,
        })

out = {"title": title, "matches": matches, "source": "the-odds-api (de-vigged, Multi-Buchmacher-Mittel)"}
json.dump(out, open("data/market.json", "w"), ensure_ascii=False, indent=1)
print(f"  market.json: {len(title)} Titel-Quoten, {len(matches)} Spiele")
top = sorted(title.items(), key=lambda x: -x[1])[:8]
print("  Markt-Titel-Top-8:", " · ".join(f"{n} {p*100:.0f}%" for n, p in top))
