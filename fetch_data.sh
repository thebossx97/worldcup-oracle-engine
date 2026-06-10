#!/usr/bin/env bash
# Lädt die historische Länderspiel-Datenbasis (martj42 international_results).
mkdir -p data
curl -sSL -o data/results.csv "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
echo "results.csv: $(wc -l < data/results.csv) Zeilen"
