"""ML v2 — Kandidaten-Features, sauberer Split (Council): Train<2015 / Val 2015-2017 / Test 2018+.
Jedes Feature EINZELN auf Validierung getestet, nur behalten was hilft, dann EINMAL final auf Test.
Kein Test-Leakage, kein Multiple-Testing am Endsatz."""
import csv, math
from datetime import datetime
from collections import defaultdict, deque
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, brier_score_loss

def pdate(s):
    try: return datetime.strptime(s.strip(), "%Y-%m-%d")
    except Exception: return None

def build():
    M = []
    with open("data/results.csv") as f:
        for r in csv.DictReader(f):
            if r["home_score"] in ("", "NA") or r["away_score"] in ("", "NA"): continue
            d = pdate(r["date"])
            if d: M.append((d, r["home_team"], r["away_team"], int(r["home_score"]), int(r["away_score"]),
                            r["neutral"] == "TRUE", r["tournament"]))
    M.sort(key=lambda x: x[0])
    slow = defaultdict(lambda: 1500.0); fast = defaultdict(lambda: 1500.0)
    gf = defaultdict(lambda: deque(maxlen=10)); ga = defaultdict(lambda: deque(maxlen=10))
    res = defaultdict(lambda: deque(maxlen=10)); last = {}; npl = defaultdict(int)
    sos = defaultdict(lambda: deque(maxlen=10))   # Stärke des Spielplans (Gegner-Elo)
    gdm = defaultdict(lambda: deque(maxlen=10))    # Tordifferenz-Momentum
    hh = defaultdict(lambda: deque(maxlen=6))      # Kopf-an-Kopf (Sicht des sortiert-ersten Teams)
    streak = defaultdict(int)                       # Sieges-/Niederlagen-Streak
    base, cand, ys, dates = [], [], [], []
    avg = lambda dq, df: sum(dq) / len(dq) if dq else df
    for (d, h, a, hs, as_, neu, tour) in M:
        comp = 0 if "Friendly" in tour else 1
        rh = min(60,(d-last[h]).days) if h in last else 30; ra = min(60,(d-last[a]).days) if a in last else 30
        key = (h, a) if h < a else (a, h)
        hh_h = avg(hh[key], 0.5);
        if h >= a: hh_h = 1 - hh_h  # auf h-Sicht drehen
        base.append([slow[h]-slow[a], (fast[h]-slow[h])-(fast[a]-slow[a]), 0 if neu else 1, comp,
                     avg(gf[h],1.2)-avg(gf[a],1.2), avg(ga[h],1.2)-avg(ga[a],1.2),
                     avg(res[h],0.5)-avg(res[a],0.5), rh-ra, min(npl[h],30), min(npl[a],30)])
        cand.append([hh_h-0.5,                         # 10: Kopf-an-Kopf
                     avg(sos[h],1500)-avg(sos[a],1500),# 11: Spielplan-Stärke-Diff
                     avg(gdm[h],0)-avg(gdm[a],0),      # 12: Tordifferenz-Momentum-Diff
                     max(-5,min(5,streak[h]))-max(-5,min(5,streak[a]))])  # 13: Streak-Diff
        out = 0 if hs>as_ else (1 if hs==as_ else 2); ys.append(out); dates.append(d)
        # Updates
        We = 1/(1+10**(-(slow[h]-slow[a]+(0 if neu else 65))/400)); Sa = 1 if out==0 else (0.5 if out==1 else 0)
        gd = abs(hs-as_); mult = 1 if gd<=1 else (1.5 if gd==2 else 1.75+(gd-3)/8)
        for elo, kk in ((slow,24),(fast,64)): dl=kk*mult*(Sa-We); elo[h]+=dl; elo[a]-=dl
        gf[h].append(hs); ga[h].append(as_); gf[a].append(as_); ga[a].append(hs)
        res[h].append(Sa); res[a].append(1-Sa)
        sos[h].append(slow[a]); sos[a].append(slow[h]); gdm[h].append(hs-as_); gdm[a].append(as_-hs)
        hk = Sa if h < a else (1-Sa); hh[key].append(hk)
        streak[h] = streak[h]+1 if out==0 else (0 if out==1 else min(0,streak[h])-1) if False else (streak[h]+1 if out==0 else (streak[h]-1 if out==2 else 0))
        streak[a] = streak[a]+1 if out==2 else (streak[a]-1 if out==0 else 0)
        last[h]=d; last[a]=d; npl[h]+=1; npl[a]+=1
    return np.array(base), np.array(cand), np.array(ys), dates

if __name__ == "__main__":
    B, C, y, dates = build()
    yr = np.array([d.year for d in dates])
    tr = yr < 2015; va = (yr>=2015)&(yr<2018); te = yr>=2018
    names = ["Kopf-an-Kopf","Spielplan-Stärke","TorDiff-Momentum","Streak"]
    def fit_ll(cols, mask_tr, mask_ev):
        X = np.hstack([B] + ([C[:, cols]] if cols else []))
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=4,
                                             l2_regularization=1.0, min_samples_leaf=50, random_state=0)
        clf.fit(X[mask_tr], y[mask_tr])
        return log_loss(y[mask_ev], clf.predict_proba(X[mask_ev]), labels=[0,1,2])
    print(f"  Split: Train {int(tr.sum())} (<2015) | Val {int(va.sum())} (15-17) | Test {int(te.sum())} (18+)\n")
    base_val = fit_ll([], tr, va)
    print(f"  Baseline (10 Feat)            Val-LogLoss {base_val:.4f}")
    keep = []
    for i, nm in enumerate(names):
        v = fit_ll([i], tr, va)
        better = v < base_val - 0.0002
        print(f"  + {nm:22} Val {v:.4f}  (Δ {v-base_val:+.4f}){'  ← behalten ✓' if better else ''}")
        if better: keep.append(i)
    print(f"\n  Behaltene Features: {[names[i] for i in keep] or 'KEINE — Baseline bleibt'}")
    # Kombination der behaltenen, final EINMAL auf Test
    base_test = fit_ll([], tr|va, te)
    comb_test = fit_ll(keep, tr|va, te) if keep else base_test
    print(f"\n  FINAL auf Test 2018+ (Train = <2018):")
    print(f"    Baseline 10 Feat : {base_test:.4f}")
    print(f"    + behaltene Feat : {comb_test:.4f}  (Δ {comb_test-base_test:+.4f})  {'BESSER ✓' if comb_test<base_test-0.0003 else 'kein echter Gewinn'}")
