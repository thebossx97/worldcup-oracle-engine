"""Wie gut sind unsere WM-Vorhersagen? Modell-W/U/N je Gruppenspiel vs. Markt (de-vigged, scharfe Referenz).
Misst Abweichung, Korrelation, Überkonfidenz. (Spiele noch nicht gespielt → vs Markt, nicht vs Ergebnis.)"""
import json, math

ELO = {
 'Mexico':1868,'South Africa':1700,'South Korea':1790,'Czech Republic':1810,
 'Canada':1765,'Bosnia & Herzegovina':1685,'Qatar':1610,'Switzerland':1894,
 'Brazil':1988,'Morocco':1855,'Haiti':1565,'Scotland':1800,
 'USA':1820,'Paraguay':1715,'Australia':1720,'Turkey':1906,
 'Germany':1925,'Curaçao':1550,'Curacao':1550,'Ivory Coast':1700,'Ecuador':1935,
 'Netherlands':1961,'Japan':1906,'Sweden':1815,'Tunisia':1690,
 'Belgium':1866,'Egypt':1705,'Iran':1810,'New Zealand':1560,
 'Spain':2165,'Cape Verde':1620,'Saudi Arabia':1620,'Uruguay':1892,
 'France':2081,'Senegal':1866,'Iraq':1640,'Norway':1917,
 'Argentina':2113,'Algeria':1770,'Austria':1830,'Jordan':1630,
 'Portugal':1984,'DR Congo':1720,'Uzbekistan':1660,'Colombia':1977,
 'England':2020,'Croatia':1930,'Ghana':1700,'Panama':1660,
}
HOSTS = {'Mexico','USA','Canada'}

def ppmf(l,k):
    p=math.exp(-l)
    for i in range(1,k+1): p*=l/i
    return p
def wdl(eh,ea,HA,BASE=1.35,DIV=900,CAP=3.3,MAX=11):
    d=(eh-ea+HA)/DIV; lh=min(CAP,BASE*10**d); la=min(CAP,BASE*10**(-d))
    ph=[ppmf(lh,k) for k in range(MAX)]; pa=[ppmf(la,k) for k in range(MAX)]
    w=dr=l=0.0
    for i in range(MAX):
        for j in range(MAX):
            p=ph[i]*pa[j]
            if i>j:w+=p
            elif i==j:dr+=p
            else:l+=p
    s=w+dr+l; return [w/s,dr/s,l/s]

m=json.load(open('data/market.json'))
rows=[]
for mt in m['matches']:
    h,a=mt['home'],mt['away']
    if h not in ELO or a not in ELO: continue
    HA=65 if h in HOSTS else 0
    mod=wdl(ELO[h],ELO[a],HA)
    mk=[mt['ph'],mt['pd'],mt['pa']]
    rows.append((h,a,mod,mk))

# Metriken
mae=sum(abs(mod[k]-mk[k]) for _,_,mod,mk in rows for k in range(3))/(len(rows)*3)
# Korrelation der favoriten-Wkt (max prob)
import statistics
mod_max=[max(mod) for _,_,mod,_ in rows]; mk_max=[max(mk) for _,_,_,mk in rows]
try: corr=statistics.correlation(mod_max,mk_max)
except Exception: corr=float('nan')
# Cross-Entropy: Markt als Referenz vs Modell (wie gut trifft Modell die scharfe Verteilung)
ce_mod=-sum(mk[k]*math.log(max(1e-9,mod[k])) for _,_,mod,mk in rows for k in range(3))/len(rows)
ce_self=-sum(mk[k]*math.log(max(1e-9,mk[k])) for _,_,_,mk in rows for k in range(3))/len(rows)
# Überkonfidenz: mittlere Top-Wkt
print(f"  Verglichene Gruppenspiele: {len(rows)}")
print(f"\n  WIE NAH AM MARKT (scharfe Referenz):")
print(f"    Mittlere Abweichung je Outcome (MAE): {mae*100:.1f} Prozentpunkte")
print(f"    Korrelation Favoriten-Wahrscheinlichkeit: {corr:.3f}")
print(f"    Cross-Entropy Modell vs Markt: {ce_mod:.4f}  (Markt-Eigenentropie {ce_self:.4f}, Lücke {ce_mod-ce_self:.4f})")
print(f"\n  ÜBERKONFIDENZ-CHECK:")
print(f"    Ø Favoriten-Wkt MODELL: {statistics.mean(mod_max)*100:.1f}%   MARKT: {statistics.mean(mk_max)*100:.1f}%")
# größte Abweichungen
diffs=sorted(rows,key=lambda r:-abs(max(r[2])-max(r[3])))[:4]
print(f"\n  Größte Modell-vs-Markt-Differenzen (Top-Outcome):")
for h,a,mod,mk in diffs:
    print(f"    {h} v {a}: Modell {max(mod)*100:.0f}% / Markt {max(mk)*100:.0f}%")
