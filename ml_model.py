"""Echtes ML statt Heuristik: HistGradientBoosting auf engineerte Features (Walk-Forward, kein Look-ahead).
Vergleich out-of-sample gegen Elo->Poisson-Baseline + naive Baseline (Log-Loss/Brier).
Disziplin: ML wird nur 'besser' genannt, wenn es die Elo-Baseline auf ungesehenen Daten schlaegt."""
import csv, math
from datetime import datetime
from collections import defaultdict, deque
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, brier_score_loss

def pdate(s):
    for f in ("%Y-%m-%d",):
        try: return datetime.strptime(s.strip(), f)
        except Exception: pass
    return None

# ── Elo->Poisson Baseline (gleich wie im Oracle) ──
def ppmf(l, k):
    p = math.exp(-l)
    for i in range(1, k + 1): p *= l / i
    return p
def elo_wdl(eh, ea, neutral, HA=65, BASE=1.35, DIV=900, CAP=3.3, MAX=10):
    d = (eh - ea + (0 if neutral else HA)) / DIV
    lh = min(CAP, BASE * 10 ** d); la = min(CAP, BASE * 10 ** (-d))
    ph = [ppmf(lh, k) for k in range(MAX)]; pa = [ppmf(la, k) for k in range(MAX)]
    w = dr = l = 0.0
    for i in range(MAX):
        for j in range(MAX):
            p = ph[i] * pa[j]
            if i > j: w += p
            elif i == j: dr += p
            else: l += p
    s = w + dr + l; return [w / s, dr / s, l / s]

# ── Walk-Forward Feature-Engineering ──
def build():
    M = []
    with open("data/results.csv") as f:
        for r in csv.DictReader(f):
            if r["home_score"] in ("", "NA") or r["away_score"] in ("", "NA"): continue
            d = pdate(r["date"])
            if not d: continue
            M.append((d, r["home_team"], r["away_team"], int(r["home_score"]), int(r["away_score"]),
                      r["neutral"] == "TRUE", r["tournament"]))
    M.sort(key=lambda x: x[0])

    slow = defaultdict(lambda: 1500.0); fast = defaultdict(lambda: 1500.0)
    gf = defaultdict(lambda: deque(maxlen=10)); ga = defaultdict(lambda: deque(maxlen=10))
    res = defaultdict(lambda: deque(maxlen=10))  # 1 win .5 draw 0 loss
    last = {}; nplayed = defaultdict(int)

    feats, ys, dates, elo_base = [], [], [], []
    for (d, h, a, hs, as_, neu, tour) in M:
        eh, ea = slow[h], slow[a]
        comp = 0 if "Friendly" in tour else 1
        def avg(dq, default): return sum(dq) / len(dq) if dq else default
        resth = min(60, (d - last[h]).days) if h in last else 30
        resta = min(60, (d - last[a]).days) if a in last else 30
        row = [
            eh - ea,                                   # Elo-Differenz
            (fast[h] - slow[h]) - (fast[a] - slow[a]), # Form-Differenz
            0 if neu else 1,                           # Heimvorteil
            comp,                                      # Wettbewerb vs Freundschaft
            avg(gf[h], 1.2) - avg(gf[a], 1.2),         # Tore-fuer-Rate Diff
            avg(ga[h], 1.2) - avg(ga[a], 1.2),         # Tore-gegen-Rate Diff
            avg(res[h], 0.5) - avg(res[a], 0.5),       # juengste Ergebnis-Rate Diff
            resth - resta,                             # Ruhetage-Diff
            min(nplayed[h], 30), min(nplayed[a], 30),  # Daten-Vertrauen
        ]
        out = 0 if hs > as_ else (1 if hs == as_ else 2)
        feats.append(row); ys.append(out); dates.append(d)
        elo_base.append(elo_wdl(eh, ea, neu))
        # Updates (nach dem Spiel)
        We = 1 / (1 + 10 ** (-(eh - ea + (0 if neu else 65)) / 400))
        Sa = 1 if out == 0 else (0.5 if out == 1 else 0)
        gd = abs(hs - as_); mult = 1 if gd <= 1 else (1.5 if gd == 2 else 1.75 + (gd - 3) / 8)
        for elo, kk in ((slow, 24), (fast, 64)):
            dl = kk * mult * (Sa - We); elo[h] += dl; elo[a] -= dl
        gf[h].append(hs); ga[h].append(as_); gf[a].append(as_); ga[a].append(hs)
        res[h].append(Sa); res[a].append(1 - Sa)
        last[h] = d; last[a] = d; nplayed[h] += 1; nplayed[a] += 1
    return np.array(feats), np.array(ys), dates, np.array(elo_base)

if __name__ == "__main__":
    X, y, dates, elo_base = build()
    cut = datetime(2018, 1, 1)
    tr = np.array([d < cut for d in dates]); te = ~tr
    print(f"  Spiele: {len(y)}  | Train {tr.sum()} (<2018)  Test {te.sum()} (2018+)")

    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=4,
                                         l2_regularization=1.0, min_samples_leaf=50, random_state=0)
    clf.fit(X[tr], y[tr])
    p_ml = clf.predict_proba(X[te])

    base = [sum(y[tr] == k) / tr.sum() for k in range(3)]
    p_naive = np.tile(base, (te.sum(), 1))

    yte = y[te]
    ll_ml = log_loss(yte, p_ml, labels=[0, 1, 2])
    ll_elo = log_loss(yte, elo_base[te], labels=[0, 1, 2])
    ll_naive = log_loss(yte, p_naive, labels=[0, 1, 2])
    print(f"\n  HELD-OUT 2018+ (Log-Loss, niedriger = besser):")
    print(f"    Naive Baseline   : {ll_naive:.4f}")
    print(f"    Elo->Poisson     : {ll_elo:.4f}")
    print(f"    ML (HistGB)      : {ll_ml:.4f}")
    print(f"\n    ML vs Elo: {(ll_elo-ll_ml):+.4f}  →  {'ML SCHLAEGT Elo ✓' if ll_ml<ll_elo else 'Elo bleibt besser ✗'}")
    print(f"    ML vs naive: {(1-ll_ml/ll_naive)*100:.1f}% besser")
    # Feature-Wichtigkeit (Permutation-frei: via Score-Drop grob)
    names = ["Elo-Diff","Form-Diff","Heim","Wettbewerb","ToreFuer","ToreGegen","ErgebnisRate","Ruhe","NPlayedH","NPlayedA"]
    print("\n  (Modell trainiert auf", X.shape[1], "Features:", ", ".join(names) + ")")
