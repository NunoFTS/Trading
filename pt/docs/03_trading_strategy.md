# Trading Strategy — Implementation Reference

## Overview
The SU strategy is an enhanced threshold-based mean-reversion system. It goes beyond the classic "enter when spread crosses 2σ" by adding:
1. **Momentum filter** — prevents trading against the trend
2. **Simple Reversion Test** — requires confirmation before entry
3. **Adaptive exit** — dynamically adjusts the close threshold to maximise profit

---

## Module: `src/strategy.py`

Responsible for: determining whether to enter or exit a trade on a given day, given the current z-score, momentum, and trade state.

---

## Rule 1: Momentum Filter

**Purpose:** Avoid entering trades that fight the prevailing trend of a stock.

**Definition (30-day SMA slope):**
- If `price_A[-1] > sma_30_A[-1]` → A is bullish → **block long A / short B** (block short-spread entry)
- If `price_A[-1] < sma_30_A[-1]` → A is bearish → **block long A** (block long-spread entry)

Symmetrically applied to stock B.

**Implementation:**
```python
def momentum_allows_long(price_a, price_b, sma_a, sma_b, idx):
    # Long spread = buy A, sell B
    # Block if A is bearish or B is bullish
    a_bearish = price_a[idx] < sma_a[idx]
    b_bullish  = price_b[idx] > sma_b[idx]
    return not (a_bearish or b_bullish)

def momentum_allows_short(price_a, price_b, sma_a, sma_b, idx):
    # Short spread = sell A, buy B
    # Block if A is bullish or B is bearish
    a_bullish  = price_a[idx] > sma_a[idx]
    b_bearish  = price_b[idx] < sma_b[idx]
    return not (a_bullish or b_bearish)
```

**Note:** This check happens **before** the reversion test. If momentum blocks the trade, skip entirely regardless of z-score.

---

## Rule 2: Simple Reversion Test (Entry Confirmation)

**Purpose:** Avoid entering too early when the spread is still diverging. Wait for evidence of reversal.

**How it works:**
- Track a state variable `crossed_once` per potential trade.
- On the **first** threshold breach: set `crossed_once = True`. Do NOT enter.
- On the **second** threshold breach (after the spread pulls back even slightly and crosses again): enter the trade.

**State machine per direction:**
```
State: IDLE
  → z > open_long threshold  →  State: WAITING_LONG
State: WAITING_LONG
  → z drops below threshold  →  State: IDLE  (reset, false alarm)
  → z > open_long threshold again  →  ENTER LONG trade, State: IN_TRADE
```

**Key:** The spread must pull back below the threshold between the two crossings. If it stays above continuously, that is one crossing, not two.

---

## Rule 3: Entry Thresholds

From the chromosome (Genes 3–4):

| Signal | Condition | Action |
|--------|-----------|--------|
| Long spread | `z > +open_mult_long · σ` | Buy A, Sell B |
| Short spread | `z < -open_mult_short · σ` | Sell A, Buy B |

Note: `σ` here is the rolling std from the chromosome's window. The z-score IS `S/σ` so the threshold is just the multiplier itself.

---

## Rule 4: Adaptive Exit

**Purpose:** Don't exit too early. Let winning trades run by shifting the close threshold further from zero.

**Initial close thresholds (Genes 5–6):**
- Long exit: `z < +close_mult_long`
- Short exit: `z > -close_mult_short`

**Adaptive adjustment:**
- Each time the z-score crosses the close threshold without triggering the close (i.e., touches but retreats), shift the threshold closer to zero by `exit_step` (Gene 7).
- Formally: `close_threshold_long -= exit_step` after each crossing.
- This allows the trade to capture more of the reversion before exiting.

**Floor:** The close threshold should not go below 0 (never require spread to cross to opposite side to exit).

---

## Rule 5: Stop-Loss (Gene 9 — DORMANT in Stage 1)

Gene 9 encodes a stop-loss multiplier. In Stage 1, this gene exists in the chromosome but the stop-loss logic is **not applied**. It is activated in Stage 2.

When active (Stage 2+):
- If `z > open_mult_long + stop_loss_mult` → spread has diverged too far → exit long immediately at a loss.
- Symmetric for short.

---

## Module: `src/backtest.py`

### Function: `run_backtest(spread_smooth, z_score, price_a, price_b, params, sma_a, sma_b)`

**Inputs:**
- `spread_smooth` — EMA-smoothed spread series
- `z_score` — rolling z-score of smoothed spread
- `price_a`, `price_b` — original price series (for momentum filter)
- `sma_a`, `sma_b` — 30-day SMA series for A and B
- `params` — dict with keys: `open_long`, `open_short`, `close_long`, `close_short`, `exit_step`, `window`

**Output:** List of trade dicts:
```python
{
  'entry_date': date,
  'exit_date': date,
  'direction': 'long' | 'short',
  'entry_z': float,
  'exit_z': float,
  'pnl': float,          # spread at exit - spread at entry (long) or reverse (short)
  'duration': int        # trading days held
}
```

**Historical ROI Calculation:**
```python
roi = sum(t['pnl'] for t in trades) / abs(spread_smooth.iloc[0])
```
Using initial spread magnitude as a normaliser.

---

## Important Implementation Notes

1. **One trade at a time per pair.** Do not open a new trade while one is already open.
2. **NaN handling:** Skip any timestep where `z_score` is NaN (warm-up period from rolling window).
3. **No lookahead:** All decisions at time `t` use only data up to `t`.
4. **Max trade duration:** Not explicitly capped in Stage 1. Trades can stay open indefinitely until the exit condition is met.
5. **PnL sign convention:**
   - Long trade: profit if spread falls toward mean → `pnl = spread_entry - spread_exit` (wait, check: long = buy A, sell B → profit if A rises or B falls → spread rises then falls back → entry at high z, exit at low z → `pnl = entry_spread - exit_spread`)
   - Short trade: profit if spread rises toward mean → `pnl = exit_spread - entry_spread`

---

## Verification Checklist
- [ ] No trade opens without passing the momentum filter
- [ ] Trades only open on the second confirmed crossing (reversion test)
- [ ] Adaptive exit threshold decreases over the life of a trade
- [ ] Stop-loss gene exists in params dict but has no effect on trade outcomes
- [ ] No NaN z-scores used in any decision
- [ ] PnL signs are correct (profitable trades return positive pnl)
