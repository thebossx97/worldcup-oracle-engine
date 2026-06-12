"""Der entscheidende ehrliche Test: Form-Modell auf WENIGER EFFIZIENTEN Ligen (Unterligen/kleine Maerkte).
Top-5-Ligen sind die schaersten -> kein Edge. Hier sind die Linien softer -> wenn irgendwo, dann hier.
Pro Liga UND gesamt: ROI + CLV (Value@MaxH, vs Pinnacle-Close)."""
import glob, csv, os
from collections import defaultdict
from datetime import datetime
from soft_clv_features import ppmf, wdl

def pdate(s):
    for f in ("%d/%m/%Y", "%d/%m/%y"):
        try: return datetime.strptime(s.strip(), f)
        except Exception: pass
    return None
def fl(r, k):
    try: return float(r[k])
    except Exception: return None

def load(folder):
    rows = []
    for path in glob.glob(f"data/{folder}/*.csv"):
        div = os.path.basename(path).split("_")[1].replace(".csv", "")
        with open(path, encoding="latin-1") as f:
            for r in csv.DictReader(f):
                d = pdate(r.get("Date", ""))
                try: hg, ag = int(r["FTHG"]), int(r["FTAG"])
                except Exception: continue
                psc = [fl(r, "PSCH"), fl(r, "PSCD"), fl(r, "PSCA")]
                mx = [fl(r, "MaxH"), fl(r, "MaxD"), fl(r, "MaxA")]
                if not d or None in psc or None in mx or min(psc) <= 1: continue
                rows.append((d, div, r["HomeTeam"], r["AwayTeam"], hg, ag, psc, mx))
    rows.sort(key=lambda x: x[0]); return rows

def backtest(rows, HA=90, K=20, KF=60, FORM_W=0.35, cutoff=2023, edge=0.05, by_league=False):
    slow = defaultdict(lambda: 1500.0); fast = defaultdict(lambda: 1500.0)
    agg = defaultdict(lambda: [0, 0.0, 0.0, 0.0])  # league -> [bets, stake, ret, clv]
    for (d, div, h, a, hg, ag, psc, mx) in rows:
        kh, ka = (div, h), (div, a)
        eh = slow[kh] + FORM_W * (fast[kh] - slow[kh])
        ea = slow[ka] + FORM_W * (fast[ka] - slow[ka])
        m = wdl(eh, ea, HA)
        out = 0 if hg > ag else (1 if hg == ag else 2)
        if d.year >= cutoff:
            for k in range(3):
                if m[k] * mx[k] - 1 > edge:
                    key = div if by_league else "ALL"
                    s = agg[key]; s[0] += 1; s[1] += 1; s[2] += mx[k] if out == k else 0; s[3] += mx[k] / psc[k] - 1
                    agg["ALL"][0] += 1 if by_league else 0  # ALL separat unten
        We = 1 / (1 + 10 ** (-(slow[kh] - slow[ka] + HA) / 400))
        Sa = 1 if out == 0 else (0.5 if out == 1 else 0)
        gd = abs(hg - ag); mult = 1 if gd <= 1 else (1.5 if gd == 2 else 1.75 + (gd - 3) / 8)
        for elo, kk in ((slow, K), (fast, KF)):
            dl = kk * mult * (Sa - We); elo[kh] += dl; elo[ka] -= dl
    return agg

if __name__ == "__main__":
    rows = load("soft")
    print(f"  {len(rows)} Spiele (softere Ligen) | Held-out 2023+ | Form-Modell, Value@MaxH\n")
    # Gesamt
    g = backtest(rows, by_league=False)["ALL"]
    bets, stake, ret, clv = g
    roi = (ret - stake) / stake * 100
    print(f"  GESAMT (alle softeren Ligen): {int(bets)} Wetten  |  ROI {roi:+.1f}%  |  CLV {clv/bets*100:+.2f}%")
    print(f"  (Vergleich Top-5-Ligen war: ROI -1.0% / CLV +1.3%)\n")
    print("  PRO LIGA (wo ist die Linie am softesten?):")
    per = backtest(rows, by_league=True)
    rowsout = []
    for lg, (b, st, rt, cv) in per.items():
        if lg == "ALL" or b < 80: continue
        rowsout.append((lg, (rt - st) / st * 100, cv / b * 100, int(b)))
    for lg, roi, cv, b in sorted(rowsout, key=lambda x: -x[1]):
        flag = "  ← PROFITABEL ✓" if roi > 0 else ""
        print(f"    {lg:<4} ROI {roi:+6.1f}%  CLV {cv:+.2f}%  [{b} Wetten]{flag}")
