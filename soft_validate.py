"""Validierung der Kandidaten (E2, SC3): ist der Profit ECHT oder Zufall/Data-Mining?
Test 1: pro Saison getrennt (2023-24 vs 2024-25) — hält der Edge in BEIDEN? (Out-of-sample-Konsistenz)
Test 2: Bootstrap-Konfidenzintervall auf den ROI — schließt es 0 aus?
Ehrliche Hürde: nur wenn beides haelt, glauben wir an einen Edge."""
import glob, csv, os, random
from collections import defaultdict
from datetime import datetime
from soft_clv_features import wdl

def pdate(s):
    for f in ("%d/%m/%Y", "%d/%m/%y"):
        try: return datetime.strptime(s.strip(), f)
        except Exception: pass
    return None
def fl(r, k):
    try: return float(r[k])
    except Exception: return None

def load(folder, leagues):
    rows = []
    for path in glob.glob(f"data/{folder}/*.csv"):
        div = os.path.basename(path).split("_")[1].replace(".csv", "")
        if div not in leagues: continue
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

def collect_bets(rows, league, HA=90, K=20, KF=60, FORM_W=0.35, edge=0.05):
    slow = defaultdict(lambda: 1500.0); fast = defaultdict(lambda: 1500.0)
    bets = []  # (jahr, profit)
    for (d, div, h, a, hg, ag, psc, mx) in rows:
        kh, ka = (div, h), (div, a)
        eh = slow[kh] + FORM_W * (fast[kh] - slow[kh]); ea = slow[ka] + FORM_W * (fast[ka] - slow[ka])
        m = wdl(eh, ea, HA)
        out = 0 if hg > ag else (1 if hg == ag else 2)
        if div == league and d.year >= 2023:
            for k in range(3):
                if m[k] * mx[k] - 1 > edge:
                    # Saison: Spiel ab Juli zählt zur nächsten Saison
                    yr = d.year if d.month >= 7 else d.year - 1
                    bets.append((yr, (mx[k] - 1) if out == k else -1.0))
        We = 1 / (1 + 10 ** (-(slow[kh] - slow[ka] + HA) / 400))
        Sa = 1 if out == 0 else (0.5 if out == 1 else 0)
        gd = abs(hg - ag); mult = 1 if gd <= 1 else (1.5 if gd == 2 else 1.75 + (gd - 3) / 8)
        for elo, kk in ((slow, K), (fast, KF)):
            dl = kk * mult * (Sa - We); elo[kh] += dl; elo[ka] -= dl
    return bets

def roi(profits):
    return sum(profits) / len(profits) * 100 if profits else 0

def bootstrap_ci(profits, n=2000):
    random.seed(42)
    rois = []
    for _ in range(n):
        s = [profits[random.randrange(len(profits))] for _ in range(len(profits))]
        rois.append(roi(s))
    rois.sort()
    return rois[int(0.025 * n)], rois[int(0.975 * n)]

if __name__ == "__main__":
    rows = load("soft", {"E2", "SC3"})
    for lg in ("E2", "SC3"):
        bets = collect_bets(rows, lg)
        by_season = defaultdict(list)
        for yr, p in bets: by_season[yr].append(p)
        allp = [p for _, p in bets]
        lo, hi = bootstrap_ci(allp)
        print(f"\n  === {lg} ===  ({len(bets)} Wetten gesamt)")
        print(f"    Gesamt-ROI: {roi(allp):+.1f}%   95%-Bootstrap-KI: [{lo:+.1f}%, {hi:+.1f}%]")
        sig = "SIGNIFIKANT (KI > 0) ✓" if lo > 0 else "NICHT signifikant (KI enthält 0)"
        print(f"    → {sig}")
        print(f"    Pro Saison (Konsistenz-Check):")
        for yr in sorted(by_season):
            ps = by_season[yr]
            print(f"       Saison {yr}/{(yr+1)%100:02d}: ROI {roi(ps):+6.1f}%  ({len(ps)} Wetten)")
