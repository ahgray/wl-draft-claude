# NFL Draft Optimizer

A real-time decision support tool for fantasy football snake drafts where you draft entire NFL teams (not players) to optimize for either most total wins OR most total losses.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test the system:**
   ```bash
   python test_system.py
   ```

3. **Launch the draft app:**
   ```bash
   streamlit run draft_app.py
   ```

## Features

### Core Capabilities
- **Multi-source data integration:** Massey ratings, Vegas odds, historical performance
- **Monte Carlo simulation:** 50,000+ season simulations for accurate projections
- **Real-time strategy adaptation:** Pivots between win/loss strategies based on draft flow
- **Opponent modeling:** Tracks and predicts opponent strategies
- **Position #6 optimization:** Tailored for snake draft position 6 (picks 6, 11, 22, 27)

### Key Strategies

1. **Win Maximizer:** Target teams with highest expected wins
2. **Loss Maximizer:** Target teams with highest expected losses  
3. **Chaos Strategy:** High-variance teams that could win either prize
4. **Blocking:** Deny key teams to opponents
5. **Pivot:** Switch strategies mid-draft based on opportunities

### Draft Intelligence
- **Portfolio optimization:** Minimizes correlation, maximizes expected value
- **Scarcity analysis:** Identifies when to reach vs wait
- **Value identification:** Spots teams available later than expected
- **Head-to-head tracking:** Avoids cannibalization in your portfolio

## Usage During Draft

### Pre-Draft (Days Before)
1. Run `python test_system.py` to cache all data
2. Review team rankings and simulation results
3. Identify target teams for rounds 1-2
4. Plan contingency strategies

### Draft Day
1. Launch app: `streamlit run draft_app.py`
2. Record each pick as it happens
3. Follow recommendations for your picks
4. Monitor opponent strategies
5. Use blocking when appropriate

### Your Picks (Position #6)
- **Pick #6 (Round 1):** Best available from tier 2 (likely DET, LAR, or TB)
- **Pick #11 (Round 2):** Commit to strategy (continue wins or pivot to losses)
- **Pick #22 (Round 3):** Value pick or blocking opportunity
- **Pick #27 (Round 4):** Complete portfolio with best available

## File Structure

```
├── draft_app.py              # Main Streamlit UI
├── test_system.py            # System test script
├── requirements.txt          # Python dependencies
├── massey-2025-export.csv    # Massey ratings data
│
├── src/
│   ├── data/
│   │   └── data_fetcher.py  # Data loading and caching
│   ├── simulation/
│   │   ├── win_probability.py  # Win probability models
│   │   └── monte_carlo.py      # Season simulation engine
│   └── optimization/
│       ├── draft_strategy.py   # Strategy recommendation engine
│       └── draft_tracker.py    # Real-time draft state management
│
└── data/
    └── cache/               # Cached simulation results
```

## Key Metrics Explained

- **Expected Wins/Losses:** Mean from simulations
- **P90 Wins/Losses:** 90th percentile (upside potential)
- **Playoff Prob:** Chance of 10+ wins
- **Top Pick Prob:** Chance of 3 or fewer wins
- **Value Score:** Combined metric for draft value

## Tips for Success

1. **Don't show your hand early** - Keep opponents guessing
2. **Watch for runs** - If 2+ drafters take bad teams, pivot to wins
3. **Block sparingly** - Only when opponent is clear threat
4. **Trust the simulations** - They account for schedule strength
5. **Stay flexible** - Best strategy depends on what others do

## Troubleshooting

- **"No cached data found"**: Run `python test_system.py` first
- **Slow performance**: Reduce simulations in `monte_carlo.py`
- **Missing teams**: Check team abbreviations match NFL standards

## Advanced Features

- Save/load draft state for practice runs
- Export results for post-draft analysis  
- Adjust strategy weights in real-time
- Override recommendations when needed

Good luck with your draft! 🏈