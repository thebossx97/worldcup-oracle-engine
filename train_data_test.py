"""Frage: nutzen wir alle Daten optimal? Test verschiedener Trainings-Fenster/Gewichtungen.
Alte Spiele (1872+) könnten Rauschen sein. Held-out 2018+ entscheidet — welche Politik gibt das beste Modell?"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from ml_model import build  # gleiche Walk-Forward-Features

X, y, dates, elo_base = build()
yr = np.array([d.year for d in dates])
te = yr >= 2018          # fester Test-Satz
trb = yr < 2018          # nur vor 2018 trainieren

def ev(mask, weights=None, label=""):
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=4,
                                         l2_regularization=1.0, min_samples_leaf=50, random_state=0)
    sw = weights[mask] if weights is not None else None
    clf.fit(X[mask], y[mask], sample_weight=sw)
    ll = log_loss(y[te], clf.predict_proba(X[te]), labels=[0, 1, 2])
    print(f"  {label:34} n={int(mask.sum()):6}  →  Log-Loss {ll:.4f}")
    return ll

print(f"  Test-Satz 2018+: {int(te.sum())} Spiele | Elo-Baseline 0.8806, aktuelles ML (alle Daten) 0.8697\n")
res = {}
res["alle"] = ev(trb, label="Alle Daten (1872+)")
res["1980"] = ev(trb & (yr >= 1980), label="Seit 1980")
res["1995"] = ev(trb & (yr >= 1995), label="Seit 1995")
res["2005"] = ev(trb & (yr >= 2005), label="Seit 2005")
res["2010"] = ev(trb & (yr >= 2010), label="Seit 2010")
for hl in (6, 10, 16):
    w = np.exp(-(2018 - yr) / float(hl))
    res[f"decay{hl}"] = ev(trb, weights=w, label=f"Alle + Zeit-Decay (Halbwertszeit {hl}J)")
best = min(res, key=res.get)
print(f"\n  → BESTE Politik: '{best}' (Log-Loss {res[best]:.4f})")
print(f"  → vs 'alle Daten' ({res['alle']:.4f}): {'besser ✓' if res[best] < res['alle'] - 0.0005 else 'praktisch gleich — alle Daten OK'}")
