# WorldCup Oracle — Engine 🔬

Ernsthafte Prognose-Pipeline (Phasen). Ziel: ein **validiert besser kalibriertes** Modell als naive Tipper/Elo-only — bewiesen per Out-of-Sample-Backtest (Log-Loss/Brier). **Kein** Versprechen, den Wettmarkt zu schlagen (das ist quasi unmöglich; alles andere wäre Marketing).

## Phasen
1. ✅ **Backtest-Harness + Baselines** — Walk-Forward über echte Historie (49k Spiele, 1872–2026). Elo→Poisson schlägt die naive Baseline um **16,2 %** (Log-Loss, 8.103 Held-out-Spiele ab 2018). Beste Params: HA=65, DIV=900.
2. ⏳ xG-Features (FBref/Understat) → muss Elo-Log-Loss (0.8807) schlagen
3. ⏳ Markt-Prior (the-odds-api) → stärkster Einzel-Prädiktor
4. ⏳ Ensemble + Kalibrierung
5. ⏳ LLM-Feature-Extraktion (Verletzungen/Aufstellung) + Live-Pipeline

## Nutzung
```bash
./fetch_data.sh          # Datenbasis holen
python3 backtest.py      # Backtest + Parameter-Suche
```

Disziplin: jedes neue Feature MUSS den Backtest-Log-Loss verbessern, sonst raus.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## ⚖️ CLV-Backtest — der ehrliche „besser als der Markt"-Test (Phase: Wahrheit)
Unabhängiges Elo→Poisson-Modell vs. **Pinnacle Closing Line** (schärfste Quote der Welt), 12.5k Vereins-Spiele, Held-out 2023+ (4.571 Spiele):
- Log-Loss Modell **1.0199** vs Closing **0.9644** → die Closing-Line schlägt das Modell (kein Edge).
- Value-Wetten zu Closing-Odds: **ROI −6,0 %** (≈ Buchmacher-Marge — wir zahlen nur den Spread).
- Favorite-Longshot-Bias: Markt ist über die ganze Spanne **quasi perfekt kalibriert** (impliz ≈ real).

**Fazit (ehrlich):** Mit einem simplen Modell + öffentlichen Daten schlägt man den scharfen Markt NICHT — und jetzt ist es BEWIESEN statt behauptet. Ein echter Edge bräuchte: schnellere Daten (News/Aufstellung vor der Linienbewegung), genuin neue Features, oder weniger effiziente Märkte (Unterligen/In-Play) — NICHT die WM-Hauptmärkte. Das CLV-Harness würde jeden echten Edge sofort erkennen, wenn wir je ein echtes Signal hinzufügen.
