"""CLV-Backtest — der EINZIGE ehrliche 'besser als der Markt'-Test.
Unabhaengiges Elo->Poisson-Modell vs. Pinnacle CLOSING line (die schaerfste Quote).
Misst: Log-Loss(Modell) vs Log-Loss(Closing) + ROI von Value-Wetten zu Closing-Odds + Favorite-Longshot-Bias.
Schlaegt das Modell die Closing-Line out-of-sample NICHT -> wir haben KEINEN Edge (und wissen es ehrlich)."""
import csv, glob, math, os
from collections import defaultdict

def pdate(s):
    s = s.strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            from datetime import datetime
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return None

def load():
    rows = []
    for path in glob.glob("data/clv/*.csv"):
        div = os.path.basename(path).split("_")[1].replace(".csv", "")
        with open(path, encoding="latin-1") as f:
            for r in csv.DictReader(f):
                try:
                    d = pdate(r.get("Date", ""))
                    psh, psd, psa = float(r["PSCH"]), float(r["PSCD"]), float(r["PSCA"])
                    hg, ag = int(r["FTHG"]), int(r["FTAG"])
                except Exception:
                    continue
                if not d or psh <= 1 or psd <= 1 or psa <= 1:
                    continue
                rows.append((d, div, r["HomeTeam"], r["AwayTeam"], hg, ag, psh, psd, psa))
    rows.sort(key=lambda x: x[0])
    return rows

def ppmf(lmb, k):
    p = math.exp(-lmb)
    for i in range(1, k + 1): p *= lmb / i
    return p

def wdl(eh, ea, HA, BASE=1.35, DIV=900, CAP=3.3, MAX=11):
    d = (eh - ea + HA) / DIV
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
    return [w / s, dr / s, l / s]

def devig(o):
    inv = [1 / o[0], 1 / o[1], 1 / o[2]]; s = sum(inv)
    return [x / s for x in inv]

def main(HA=90, K=20, cutoff_year=2023):
    rows = load()
    print(f"  Spiele mit Pinnacle-Closing: {len(rows)}  |  {rows[0][0].date()} … {rows[-1][0].date()}")
    elo = defaultdict(lambda: 1500.0)
    ll_m = ll_c = 0.0; n = 0
    stake = ret = 0.0; bets = 0
    flb = defaultdict(lambda: [0, 0.0])  # bucket -> [n, treffer]
    for (d, div, h, a, hg, ag, psh, psd, psa) in rows:
        eh, ea = elo[(div, h)], elo[(div, a)]
        m = wdl(eh, ea, HA)
        c = devig([psh, psd, psa])
        out = 0 if hg > ag else (1 if hg == ag else 2)
        odds = [psh, psd, psa]
        if d.year >= cutoff_year:  # Held-out
            ll_m += -math.log(max(1e-9, m[out]))
            ll_c += -math.log(max(1e-9, c[out]))
            n += 1
            # Value-Wetten: Modell sieht mehr Wert als der Markt (EV>0 lt. Modell)
            for k in range(3):
                ev = m[k] * odds[k] - 1  # erwartete Rendite lt. Modell
                if ev > 0.05:  # nur klare Value-Picks
                    stake += 1; bets += 1
                    ret += odds[k] if out == k else 0
            # Favorite-Longshot-Bias: nach Closing-impliziter Wkt bucketn (Außenseiter-Tipps)
            for k in range(3):
                b = round(c[k] * 10)  # 0..10
                flb[b][0] += 1
                flb[b][1] += 1 if out == k else 0
        # Elo-Update
        We = 1 / (1 + 10 ** (-(eh - ea + HA) / 400))
        Sa = 1 if out == 0 else (0.5 if out == 1 else 0)
        gd = abs(hg - ag); mult = 1 if gd <= 1 else (1.5 if gd == 2 else 1.75 + (gd - 3) / 8)
        delta = K * mult * (Sa - We)
        elo[(div, h)] = eh + delta; elo[(div, a)] = ea - delta
    print(f"\n  HELD-OUT ab {cutoff_year} (n={n} Spiele):")
    print(f"    Log-Loss MODELL  : {ll_m/n:.4f}")
    print(f"    Log-Loss CLOSING : {ll_c/n:.4f}   (Pinnacle, die scharfe Linie)")
    diff = ll_m/n - ll_c/n
    verdict = "MODELL schlaegt die Closing-Line ✓ (Edge!)" if diff < 0 else "Closing-Line schlaegt das Modell ✗ (kein Edge)"
    print(f"    → Diff {diff:+.4f}  →  {verdict}")
    if bets:
        roi = (ret - stake) / stake * 100
        print(f"\n  VALUE-WETTEN zu Closing-Odds: {bets} Wetten, ROI {roi:+.1f}%  ({'PROFITABEL' if roi>0 else 'Verlust'})")
    print(f"\n  FAVORITE-LONGSHOT-BIAS (Closing-implizit vs. echte Trefferquote):")
    for b in sorted(flb):
        nn, hit = flb[b]
        if nn >= 50:
            print(f"    impliz ~{b*10:>3}% → real {hit/nn*100:5.1f}%  (n={nn})")

if __name__ == "__main__":
    main()
