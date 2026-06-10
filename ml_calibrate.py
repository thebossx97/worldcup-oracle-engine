"""Hebel 2 (Council): bringt Kalibrierung (Isotonic/Sigmoid) was? Nur behalten wenn Brier+LogLoss sinken.
Auf dem v2-Feature-Modell, Test 2018+."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from ml_features_v2 import build

B, C, y, dates = build()
keep = [1, 3]  # Spielplan-Stärke, Streak (aus v2-Validierung)
X = np.hstack([B, C[:, keep]])
yr = np.array([d.year for d in dates]); tr = yr < 2018; te = yr >= 2018

def mbrier(yt, p):
    oh = np.eye(3)[yt]; return np.mean(np.sum((p - oh) ** 2, axis=1))

base = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=4,
                                      l2_regularization=1.0, min_samples_leaf=50, random_state=0)
base.fit(X[tr], y[tr]); p0 = base.predict_proba(X[te])
print(f"  Unkalibriert     LogLoss {log_loss(y[te],p0,labels=[0,1,2]):.4f}  Brier {mbrier(y[te],p0):.4f}")
for method in ("isotonic", "sigmoid"):
    cal = CalibratedClassifierCV(base, method=method, cv=3)
    cal.fit(X[tr], y[tr]); pc = cal.predict_proba(X[te])
    ll = log_loss(y[te], pc, labels=[0,1,2]); br = mbrier(y[te], pc)
    print(f"  Kalibriert {method:9} LogLoss {ll:.4f}  Brier {br:.4f}  → {'BESSER ✓' if ll<log_loss(y[te],p0,labels=[0,1,2])-0.0003 else 'kein Gewinn'}")
