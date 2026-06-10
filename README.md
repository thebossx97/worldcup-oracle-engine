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
