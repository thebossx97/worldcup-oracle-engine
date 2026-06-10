"""Features-first (Council): Form + Ruhetage hinzufügen, jeweils ΔCLV messen (vs MaxH frühe Quote).
Walk-forward, kein Look-ahead (Features nutzen nur Daten VOR dem Spiel). Hypothese: frühe Linie unterpreist
Form/Ruhe → das bringt CLV (während Dixon-Coles nur das Tor-Modell schärft, das die Linie schon kann)."""
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
                if not d or None in psc or None in mx: continue
                rows.append((d, div, r["HomeTeam"], r["AwayTeam"], hg, ag, psc, mx))
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

def run(rows, HA=90, K=20, KF=60, FORM_W=0.0, REST_W=0.0, cutoff=2023, edge=0.05):
    slow = defaultdict(lambda: 1500.0); fast = defaultdict(lambda: 1500.0)
    last = {}
    bets = stake = ret = clv = 0.0; n = 0
    for (d, div, h, a, hg, ag, psc, mx) in rows:
        kh, ka = (div, h), (div, a)
        # Form = Abweichung schnelles vom langsamen Elo
        eh = slow[kh] + FORM_W * (fast[kh] - slow[kh])
        ea = slow[ka] + FORM_W * (fast[ka] - slow[ka])
        # Ruhe: Tage seit letztem Spiel (gedeckelt), mehr Ruhe = kleiner Bonus
        ha = HA
        if REST_W and kh in last and ka in last:
            rh = min(14, (d - last[kh]).days); ra = min(14, (d - last[ka]).days)
            ha += REST_W * (rh - ra)
        m = wdl(eh, ea, ha)
        out = 0 if hg > ag else (1 if hg == ag else 2)
        if d.year >= cutoff:
            n += 1
            for k in range(3):
                if m[k] * mx[k] - 1 > edge:
                    bets += 1; stake += 1
                    ret += mx[k] if out == k else 0
                    clv += (mx[k] / psc[k] - 1)
        # Updates (nach dem Spiel → kein Look-ahead)
        We = 1 / (1 + 10 ** (-(slow[kh] - slow[ka] + HA) / 400))
        Sa = 1 if out == 0 else (0.5 if out == 1 else 0)
        gd = abs(hg - ag); mult = 1 if gd <= 1 else (1.5 if gd == 2 else 1.75 + (gd - 3) / 8)
        for elo, kk in ((slow, K), (fast, KF)):
            dl = kk * mult * (Sa - We)
            elo[kh] += dl; elo[ka] -= dl
        last[kh] = d; last[ka] = d
    roi = (ret - stake) / stake * 100 if bets else 0
    return roi, (clv / bets * 100 if bets else 0), int(bets)

if __name__ == "__main__":
    rows = load()
    print(f"  {len(rows)} Spiele | Held-out 2023+ | Metrik: ROI + CLV (vs Pinnacle-Close), Value@MaxH\n")
    configs = [
        ("Baseline (nur Elo)", 0.0, 0.0),
        ("+ Form", 0.35, 0.0),
        ("+ Ruhetage", 0.0, 4.0),
        ("+ Form + Ruhe", 0.35, 4.0),
    ]
    base = None
    for name, fw, rw in configs:
        roi, clvv, bets = run(rows, FORM_W=fw, REST_W=rw)
        if base is None: base = clvv
        dclv = clvv - base
        flag = "  ← CLV ↑" if dclv > 0.15 else ("  ← CLV ↓" if dclv < -0.15 else "")
        print(f"  {name:<22} ROI {roi:+5.1f}%  |  CLV {clvv:+.2f}%  (Δ {dclv:+.2f}){flag}  [{bets} Wetten]")
