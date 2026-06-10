"""WM-Vorhersagen mit dem ECHTEN ML-Modell (statt Elo-Heuristik).
Trainiert HistGradientBoosting auf der Historie, scort alle WM-Paarungen, Monte-Carlo-Turnier → ml.json.
Exportiert nach DEUTSCHEN Teamnamen (fürs Frontend)."""
import csv, math, json, random
from datetime import datetime
from collections import defaultdict, deque
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

# (results.csv-Name, deutscher Name, Gruppe, kuratiertes Elo-Prior)
TEAMS = [
 ("Mexico","Mexiko","A",1868),("South Africa","Südafrika","A",1700),("South Korea","Südkorea","A",1790),("Czech Republic","Tschechien","A",1810),
 ("Canada","Kanada","B",1765),("Bosnia and Herzegovina","Bosnien-H.","B",1685),("Qatar","Katar","B",1610),("Switzerland","Schweiz","B",1894),
 ("Brazil","Brasilien","C",1988),("Morocco","Marokko","C",1855),("Haiti","Haiti","C",1565),("Scotland","Schottland","C",1800),
 ("United States","USA","D",1820),("Paraguay","Paraguay","D",1715),("Australia","Australien","D",1720),("Turkey","Türkei","D",1906),
 ("Germany","Deutschland","E",1925),("Curaçao","Curaçao","E",1550),("Ivory Coast","Elfenbeinküste","E",1700),("Ecuador","Ecuador","E",1935),
 ("Netherlands","Niederlande","F",1961),("Japan","Japan","F",1906),("Sweden","Schweden","F",1815),("Tunisia","Tunesien","F",1690),
 ("Belgium","Belgien","G",1866),("Egypt","Ägypten","G",1705),("Iran","Iran","G",1810),("New Zealand","Neuseeland","G",1560),
 ("Spain","Spanien","H",2165),("Cape Verde","Kap Verde","H",1620),("Saudi Arabia","Saudi-Arabien","H",1620),("Uruguay","Uruguay","H",1892),
 ("France","Frankreich","I",2081),("Senegal","Senegal","I",1866),("Iraq","Irak","I",1640),("Norway","Norwegen","I",1917),
 ("Argentina","Argentinien","J",2113),("Algeria","Algerien","J",1770),("Austria","Österreich","J",1830),("Jordan","Jordanien","J",1630),
 ("Portugal","Portugal","K",1984),("DR Congo","DR Kongo","K",1720),("Uzbekistan","Usbekistan","K",1660),("Colombia","Kolumbien","K",1977),
 ("England","England","L",2020),("Croatia","Kroatien","L",1930),("Ghana","Ghana","L",1700),("Panama","Panama","L",1660),
]
EN2DE = {en: de for en, de, g, e in TEAMS}
PRIOR = {en: e for en, de, g, e in TEAMS}
GROUPS = defaultdict(list)
for en, de, g, e in TEAMS: GROUPS[g].append(en)
SEED_ORDER = [1,32,16,17,8,25,9,24,4,29,13,20,5,28,12,21,2,31,15,18,7,26,10,23,3,30,14,19,6,27,11,22]

def pdate(s):
    try: return datetime.strptime(s.strip(), "%Y-%m-%d")
    except Exception: return None

def build_and_train():
    M = []
    with open("data/results.csv") as f:
        for r in csv.DictReader(f):
            if r["home_score"] in ("", "NA"): continue
            d = pdate(r["date"])
            if d: M.append((d, r["home_team"], r["away_team"], int(r["home_score"]), int(r["away_score"]),
                            r["neutral"] == "TRUE", r["tournament"]))
    M.sort(key=lambda x: x[0])
    slow = defaultdict(lambda: 1500.0); fast = defaultdict(lambda: 1500.0)
    gf = defaultdict(lambda: deque(maxlen=10)); ga = defaultdict(lambda: deque(maxlen=10))
    res = defaultdict(lambda: deque(maxlen=10)); last = {}; npl = defaultdict(int)
    X, y = [], []
    avg = lambda dq, df: sum(dq) / len(dq) if dq else df
    def feat(h, a, neu, comp, rh, ra):
        return [slow[h]-slow[a], (fast[h]-slow[h])-(fast[a]-slow[a]), 0 if neu else 1, comp,
                avg(gf[h],1.2)-avg(gf[a],1.2), avg(ga[h],1.2)-avg(ga[a],1.2),
                avg(res[h],0.5)-avg(res[a],0.5), rh-ra, min(npl[h],30), min(npl[a],30)]
    for (d, h, a, hs, as_, neu, tour) in M:
        comp = 0 if "Friendly" in tour else 1
        rh = min(60,(d-last[h]).days) if h in last else 30; ra = min(60,(d-last[a]).days) if a in last else 30
        X.append(feat(h,a,neu,comp,rh,ra)); y.append(0 if hs>as_ else (1 if hs==as_ else 2))
        We = 1/(1+10**(-(slow[h]-slow[a]+(0 if neu else 65))/400)); Sa = 1 if hs>as_ else (0.5 if hs==as_ else 0)
        gd = abs(hs-as_); mult = 1 if gd<=1 else (1.5 if gd==2 else 1.75+(gd-3)/8)
        for elo, kk in ((slow,24),(fast,64)): dl=kk*mult*(Sa-We); elo[h]+=dl; elo[a]-=dl
        gf[h].append(hs); ga[h].append(as_); gf[a].append(as_); ga[a].append(hs)
        res[h].append(Sa); res[a].append(1-Sa); last[h]=d; last[a]=d; npl[h]+=1; npl[a]+=1
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=4,
                                         l2_regularization=1.0, min_samples_leaf=50, random_state=0)
    clf.fit(np.array(X), np.array(y))
    # Seed schwacher Teams (sparse Historie) mit kuratiertem Prior, damit WM-Staerken stimmen
    for en, pr in PRIOR.items():
        if npl[en] < 8: slow[en] = pr; fast[en] = pr
    state = dict(slow=slow, fast=fast, gf=gf, ga=ga, res=res, npl=npl, avg=avg, feat=feat)
    return clf, state

def pairwise(clf, st):
    P = {}
    rows, keys = [], []
    for en1, _, _, _ in TEAMS:
        for en2, _, _, _ in TEAMS:
            if en1 == en2: continue
            rows.append(st["feat"](en1, en2, True, 1, 5, 5)); keys.append((en1, en2))
    probs = clf.predict_proba(np.array(rows))
    for (k, p) in zip(keys, probs): P[k] = p  # [W,D,L] aus Sicht en1
    return P

def sim(P, rng):
    firsts=[]; seconds=[]; thirds=[]
    for g, teams in GROUPS.items():
        tab = {t:[0,0] for t in teams}  # pts, gd
        for i in range(len(teams)):
            for j in range(i+1,len(teams)):
                a,b=teams[i],teams[j]; w,dr,l=P[(a,b)]; r=rng.random()
                if r<w: tab[a][0]+=3; g_=1+int(rng.random()*2); tab[a][1]+=g_; tab[b][1]-=g_
                elif r<w+dr: tab[a][0]+=1; tab[b][0]+=1
                else: tab[b][0]+=3; g_=1+int(rng.random()*2); tab[b][1]+=g_; tab[a][1]-=g_
        rank=sorted(teams,key=lambda t:(-tab[t][0],-tab[t][1],-PRIOR[t]))
        firsts.append((0,rank[0],tab[rank[0]])); seconds.append((1,rank[1],tab[rank[1]])); thirds.append((2,rank[2],tab[rank[2]]))
    thirds.sort(key=lambda x:(-x[2][0],-x[2][1],-PRIOR[x[1]])); best=thirds[:8]
    qual=sorted(firsts+seconds+best,key=lambda x:(x[0],-x[2][0],-x[2][1],-PRIOR[x[1]]))
    seeds=[q[1] for q in qual]
    rnd=[seeds[s-1] for s in SEED_ORDER]
    while len(rnd)>1:
        nxt=[]
        for i in range(0,len(rnd),2):
            a,b=rnd[i],rnd[i+1]; w,dr,l=P[(a,b)]; pw=w+dr*0.5  # K.o.: Remis per Coin
            nxt.append(a if rng.random()<pw/(pw+(l+dr*0.5)) else b)
        rnd=nxt
    return rnd[0]

if __name__ == "__main__":
    clf, st = build_and_train()
    P = pairwise(clf, st)
    rng = random.Random(12345); N = 5000; title = defaultdict(int)
    for _ in range(N): title[sim(P, rng)] += 1
    title_de = {EN2DE[en]: c / N for en, c in title.items()}
    # Spiel-Prognosen je Gruppenspiel (ML W/U/N)
    matches = []
    for g, teams in GROUPS.items():
        for i in range(len(teams)):
            for j in range(i+1, len(teams)):
                a, b = teams[i], teams[j]; w, dr, l = P[(a, b)]
                matches.append({"group": g, "home": EN2DE[a], "away": EN2DE[b],
                                "w": round(float(w),3), "d": round(float(dr),3), "l": round(float(l),3)})
    # Volle Paarungstabelle (deutsche Namen) — treibt die GANZE Frontend-Engine (Gruppen/Spiel/Durchlauf/MC)
    pairs = {f"{EN2DE[a]}|{EN2DE[b]}": [round(float(p[0]),4), round(float(p[1]),4), round(float(p[2]),4)]
             for (a, b), p in P.items()}
    out = {"title": {k: round(v,4) for k,v in sorted(title_de.items(), key=lambda x:-x[1])},
           "matches": matches, "pairs": pairs,
           "source": "HistGradientBoosting ML (10 Features), Monte-Carlo "+str(N)}
    json.dump(out, open("data/ml.json","w"), ensure_ascii=False, indent=1)
    top = sorted(title_de.items(), key=lambda x:-x[1])[:8]
    print("  ml.json geschrieben.")
    print("  ML-Titelchancen Top-8:", " · ".join(f"{k} {v*100:.0f}%" for k,v in top))
