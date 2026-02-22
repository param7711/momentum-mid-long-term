# Hybrid Trend + Swing Momentum (Backtrader)

Backtrader port of the Pine strategy with two modes:

1. **Original V6 signal mode** (default): single asset, behavior mirrors the prior V6 implementation.
2. **Portfolio overlay mode** (`--overlay`): keeps V6 signal logic unchanged per symbol, and adds weekly ranking, breadth-based exposure, replacement logic, and capital allocation at the portfolio layer.

## Install

```bash
pip install backtrader
```

## Data format

CSV with header order:

`Date,Open,High,Low,Close,Volume`

Date format: `YYYY-MM-DD`

## Run

### 1) Original V6 mode (single symbol)

```bash
python backtrader_hybrid_momentum.py --data RELIANCE.csv --cash 100000 --size 4 --commission 0.0
```

### 2) Overlay mode (multi-symbol universe)

```bash
python backtrader_hybrid_momentum.py \
  --overlay \
  --data INFY.csv TCS.csv HDFCBANK.csv ICICIBANK.csv NIFTY50.csv \
  --entry-rank-threshold 25 \
  --exit-rank-threshold 40 \
  --improvement-threshold 0.05 \
  --base-exposure 0.90 \
  --max-total-exposure 1.20 \
  --cash-reserve 0.10 \
  --min-position-weight 0.05 \
  --index-data-name NIFTY50
```

## Overlay design highlights

- **V6 engine preserved**: entry/exit logic and indicators are unchanged per symbol.
- **Weekly ranking**: computes a structural momentum score and ranks the universe.
- **Breadth model**: maps eligible trend count to allowed exposure.
- **Replacement flow**: replacement attempts are event-driven after V6-driven exits.
- **Opportunity-cost guard**: new candidate must beat weakest holding by `improvement_threshold`.
- **Rank hysteresis**: entry rank threshold and looser exit rank threshold.
- **Capital rules**: permanent cash reserve + equal-weight deployment across held positions.
- **MTF behavior**: margin only expands number of positions when breadth is strong.
- **Volatility safety**: market volatility throttles allowed exposure.
