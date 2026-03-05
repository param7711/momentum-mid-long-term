# Momentum Breakout Portfolio (Backtrader)

Run:

```bash
python backtrader_hybrid_momentum.py --start 2015-01-01 --end 2025-12-31 --capital 100000
```

Optional full Nifty500 symbols file:

```bash
python backtrader_hybrid_momentum.py --symbols-file nifty500_symbols.txt
```

Symbols file format: one symbol per line (`RELIANCE`, `TCS`, etc.); `.NS` suffix is auto-added.
