"""Fairer Edge-Test: Modell vs SOFTE/FRÜHE Linien (nicht nur Pinnacle-Closing).
Value-Wetten zur besten frühen Quote (MaxH = beste verfügbare) + Closing-Line-Value (CLV) gegen Pinnacle-Close.
CLV>0 = wir bekamen einen besseren Preis als die scharfe Schlussquote → echtes Edge-Signal."""
import csv, glob, math, os
from collections import defaultdict
from datetime import datetime

def pdate(s):
    for f in ("%d/%m/%Y", "%d/%m/%y"):
        try: return datetime.strptime(s.strip(), f)
        except Exception: pass
    return None

def fl(r, k):
    try: return float(r[k])
    except Exception: return None

def load():
    rows = []
    for path in glob.glob("data/clv/*.csv"):
        div = os.path.basename(path).split("_")[1].replace(".csv", "")
        with open(path, encoding="latin-1") as f:
            for r in csv.DictReader(f):
                d = pdate(r.get("Date", ""))
                try: hg, ag = int(r["FTHG"]), int(r["FTAG"])
                except Exception: continue
                psc = [fl(r, "PSCH"), fl(r, "PSCD"), fl(r, "PSCA")]
                mx = [fl(r, "MaxH"), fl(r, "MaxD"), fl(r, "MaxA")]
                av = [fl(r, "AvgH"), fl(r, "AvgD"), fl(r, "AvgA")]
                if not d or None in psc or None in mx or None in av: continue
                rows.append((d, div, r["HomeTeam"], r["AwayTeam"], hg, ag, psc, mx, av))
    rows.sort(key=lambda x: x[0]); return rows

def ppmf(l, k):
    p = math.exp(-l)
    for i in range(1, k + 1): p *= l / i
    return p
def wdl(eh, ea, HA, BASE=1.35, DIV=900, CAP=3.3, MAX=11):
    d = (eh - ea + HA) / DIV; lh = min(CAP, BASE * 10 ** d); la = min(CAP, BASE * 10 ** (-d))
    ph = [ppmf(lh, k) for k in range(MAX)]; pa = [ppmf(la, k) for k in range(MAX)]
    w = dr = l = 0.0
    for i in range(MAX):
        for j in range(MAX):
            p = ph[i] * pa[j]
            if i > j: w += p
            elif i == j: dr += p
            else: l += p
    s = w + dr + l; return [w / s, dr / s, l / s]

def run(HA=90, K=20, cutoff=2023, edge=0.05):
    rows = load()
    elo = defaultdict(lambda: 1500.0)
    res = {"Max (beste frühe)": [0, 0.0, 0.0, 0], "Avg (mittlere frühe)": [0, 0.0, 0.0, 0]}
    # [bets, stake, return, clv_sum]
    n = 0
    for (d, div, h, a, hg, ag, psc, mx, av) in rows:
        eh, ea = elo[(div, h)], elo[(div, a)]
        m = wdl(eh, ea, HA)
        out = 0 if hg > ag else (1 if hg == ag else 2)
        if d.year >= cutoff:
            n += 1
            for label, odds in (("Max (beste frühe)", mx), ("Avg (mittlere frühe)", av)):
                for k in range(3):
                    if m[k] * odds[k] - 1 > edge:  # Value lt. Modell zur frühen Quote
                        res[label][0] += 1; res[label][1] += 1
                        res[label][2] += odds[k] if out == k else 0
                        res[label][3] += (odds[k] / psc[k] - 1)  # CLV vs Pinnacle-Closing
        We = 1 / (1 + 10 ** (-(eh - ea + HA) / 400))
        Sa = 1 if out == 0 else (0.5 if out == 1 else 0)
        gd = abs(hg - ag); mult = 1 if gd <= 1 else (1.5 if gd == 2 else 1.75 + (gd - 3) / 8)
        delta = K * mult * (Sa - We)
        elo[(div, h)] = eh + delta; elo[(div, a)] = ea - delta
    print(f"  Held-out ab {cutoff}: {n} Spiele | Value-Schwelle {edge*100:.0f}%")
    for label, (bets, stake, ret, clv) in res.items():
        if bets:
            roi = (ret - stake) / stake * 100
            clvp = clv / bets * 100
            tag = "✓ PROFITABEL" if roi > 0 else "Verlust"
            print(f"\n  Quelle: {label}")
            print(f"    Value-Wetten: {bets}  |  ROI {roi:+.1f}%  ({tag})")
            print(f"    CLV (Preis vs Pinnacle-Close): {clvp:+.1f}%  {'→ Linie bewegt sich ZU uns (Edge-Signal)' if clvp>0 else '→ keine günstige Bewegung'}")

if __name__ == "__main__":
    run()
