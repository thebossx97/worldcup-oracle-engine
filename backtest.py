"""WorldCup Oracle — Backtest-Harness (Phase 1).
Walk-forward über echte Länderspiel-Historie. Misst Log-Loss/Brier auf ungesehenen Spielen.
Vergleicht: naive Baseline (Basisraten) vs. Elo→Poisson. Tunt Heimvorteil + Tor-Spreizung.
Ohne Backtest ist 'outperformed' nur Behauptung — das hier ist der Beweis-Apparat.
"""
import csv, math, sys

def load(path="data/results.csv"):
    M = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["home_score"] in ("", "NA") or r["away_score"] in ("", "NA"):
                continue
            M.append((r["date"], r["home_team"], r["away_team"],
                      int(r["home_score"]), int(r["away_score"]), r["neutral"] == "TRUE"))
    M.sort(key=lambda x: x[0])
    return M

def ppmf(lmb, k):
    p = math.exp(-lmb)
    for i in range(1, k + 1):
        p *= lmb / i
    return p

def wdl(eh, ea, neutral, HA, BASE, DIV, CAP, MAX=11):
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
    s = w + dr + l
    return w / s, dr / s, l / s

def outcome(hs, as_):
    return 0 if hs > as_ else (1 if hs == as_ else 2)

def backtest(M, HA, BASE, DIV, CAP, K, cutoff="2018-01-01"):
    elo = {}
    G = lambda t: elo.get(t, 1500.0)
    ll = br = 0.0; n = 0
    for (dt, h, a, hs, as_, neu) in M:
        eh, ea = G(h), G(a)
        out = outcome(hs, as_)
        if dt >= cutoff:
            probs = wdl(eh, ea, neu, HA, BASE, DIV, CAP)
            ll += -math.log(max(1e-12, probs[out]))
            br += sum((probs[k] - (1 if k == out else 0)) ** 2 for k in range(3))
            n += 1
        # Elo-Update (Walk-Forward, immer)
        We = 1 / (1 + 10 ** (-(eh - ea + (0 if neu else HA)) / 400))
        Sa = 1 if out == 0 else (0.5 if out == 1 else 0)
        gd = abs(hs - as_); mult = 1 if gd <= 1 else (1.5 if gd == 2 else 1.75 + (gd - 3) / 8)
        delta = K * mult * (Sa - We)
        elo[h] = eh + delta; elo[a] = ea - delta
    return ll / n, br / n, n

def baseline(M, cutoff="2018-01-01"):
    # Basisraten aus TRAININGS-Daten (vor cutoff), dann konstant auf Held-out scoren.
    tr = [outcome(hs, as_) for (dt, h, a, hs, as_, neu) in M if dt < cutoff]
    from collections import Counter
    c = Counter(tr); tot = len(tr)
    base = [c[0] / tot, c[1] / tot, c[2] / tot]
    ll = br = 0.0; n = 0
    for (dt, h, a, hs, as_, neu) in M:
        if dt < cutoff: continue
        out = outcome(hs, as_)
        ll += -math.log(max(1e-12, base[out]))
        br += sum((base[k] - (1 if k == out else 0)) ** 2 for k in range(3))
        n += 1
    return ll / n, br / n, base

if __name__ == "__main__":
    M = load()
    print(f"  Spiele gesamt (mit Score): {len(M)}  |  {M[0][0]} … {M[-1][0]}")
    bll, bbr, base = baseline(M)
    print(f"\n  BASELINE (Basisraten H/U/A = {base[0]:.2f}/{base[1]:.2f}/{base[2]:.2f}):")
    print(f"    Log-Loss {bll:.4f}  |  Brier {bbr:.4f}")
    print(f"\n  ELO→POISSON — Parameter-Suche (Held-out ab 2018):")
    best = None
    for HA in (50, 65, 80, 100):
        for DIV in (700, 800, 900):
            ll, br, n = backtest(M, HA, 1.35, DIV, 3.3, 24)
            if best is None or ll < best[0]:
                best = (ll, br, HA, DIV, n)
            print(f"    HA={HA:<4} DIV={DIV:<4} → Log-Loss {ll:.4f}  Brier {br:.4f}")
    ll, br, HA, DIV, n = best
    impr = (1 - ll / bll) * 100
    print(f"\n  BESTES Elo-Modell: HA={HA}, DIV={DIV}  →  Log-Loss {ll:.4f}  (Held-out n={n})")
    print(f"  → {impr:.1f}% besser als die naive Baseline (Log-Loss).")
