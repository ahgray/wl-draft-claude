# 🏈 NFL Draft Optimizer

A comprehensive real-time decision support tool for fantasy football snake drafts where you draft entire NFL teams (not players) to optimize for either most total wins OR most total losses.

## 🚀 Quick Start (Draft Day Ready!)

### Option 1: Automated Launch
```bash
./launch_app.sh
```

### Option 2: Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install pandas numpy scipy requests beautifulsoup4 sqlalchemy python-dotenv tqdm streamlit altair plotly

# Test the system
python simple_test.py

# Launch the app
streamlit run complete_app.py
```

### Access Your Draft Tool
Open your browser to: **http://localhost:8501**

## Features

### ✅ Verified Working Features
- **✓ Multi-source data integration:** Massey ratings, Vegas win totals, composite rankings
- **✓ Real-time draft tracking:** Live pick recording and portfolio analysis  
- **✓ Position #6 optimization:** Tailored for snake draft picks 6, 11, 22, 27
- **✓ Strategy adaptation:** Dynamic recommendations (Win/Loss/Balanced/Chaos)
- **✓ Opponent analysis:** Track competitor portfolios and strategies
- **✓ Portfolio optimization:** Win/loss projection with correlations
- **✓ Interactive UI:** Streamlit dashboard with real-time updates

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

## 🎯 Usage During Draft

### Pre-Draft Setup (5 minutes)
1. Run `./launch_app.sh` or manual setup commands
2. Verify system works with `python simple_test.py`
3. Review team rankings in the app
4. Plan your position #6 strategy

### Live Draft Usage
1. **Open browser:** http://localhost:8501
2. **Record picks:** Use sidebar to log each pick as it happens
3. **Follow recommendations:** Top pick highlighted with reasoning
4. **Monitor opponents:** Track their strategies in Draft Board tab
5. **Adapt strategy:** System recommends pivots automatically

### Your Pick Strategy (Position #6)
- **Pick #6:** Best available elite team (likely DET, LAR, TB if top 5 gone)
- **Pick #11:** Commit to direction - continue wins or pivot to losses  
- **Pick #22:** Value pick or block opponents showing clear strategy
- **Pick #27:** Complete portfolio balance or double-down

### App Interface Guide
- **📊 Recommendations Tab:** Your pick suggestions with strategy
- **🏈 Team Rankings:** Complete sortable team list
- **📈 Draft Board:** Live tracking of all drafters' picks
- **🧠 Strategy Tab:** Analysis and position-specific advice

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