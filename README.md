# 🏈 NFL Draft Optimizer

A comprehensive real-time decision support tool for fantasy football snake drafts where you draft entire NFL teams (not players) to optimize for either most total wins OR most total losses. Includes complete post-draft analysis with prize probability calculations.

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
- **✓ Universal draft position support:** Works for any position 1-8 with position-specific strategies
- **✓ Strategy adaptation:** Dynamic recommendations (Win/Loss/Balanced/Chaos)
- **✓ Opponent analysis:** Track competitor portfolios and strategies
- **✓ Portfolio optimization:** Win/loss projection with correlations
- **✓ Interactive UI:** Streamlit dashboard with real-time updates
- **✓ Post-draft analysis:** Comprehensive performance analysis with prize probabilities
- **✓ Smart snake draft tracking:** Correct pick order calculation for all rounds

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
4. **Select your draft position (1-8)** in the app sidebar

### Live Draft Usage
1. **Open browser:** http://localhost:8501
2. **Select draft position:** Choose your position 1-8 in sidebar
3. **Record picks:** Use sidebar to log each pick as it happens
4. **Follow recommendations:** Top pick highlighted with reasoning
5. **Monitor opponents:** Track their strategies in Draft Board tab
6. **Adapt strategy:** System recommends pivots automatically

### Draft Position Strategies
- **Positions 1-3:** Elite teams available, can show strategy early
- **Positions 4-6:** Maximum flexibility, balanced approach optimal
- **Positions 7-8:** Value plays, leverage back-to-back picks in even rounds

### App Interface Guide
- **📊 Recommendations Tab:** Your pick suggestions with strategy
- **🏈 Team Rankings:** Complete sortable team list
- **📈 Draft Board:** Live tracking of all drafters' picks
- **🧠 Strategy Tab:** Analysis and position-specific advice
- **🏆 Post-Draft Analysis:** Appears when all 32 teams drafted

## File Structure

```
├── complete_app.py           # Main Streamlit application
├── test_system.py            # System test script  
├── test_post_draft.py        # Post-draft analysis test
├── test_positions.py         # Snake draft position test
├── requirements.txt          # Python dependencies
├── massey-2025-export.csv    # Massey ratings data
│
├── src/
│   ├── data/
│   │   └── data_fetcher.py     # Data loading and caching
│   ├── simulation/
│   │   ├── win_probability.py  # Win probability models
│   │   └── monte_carlo.py      # Season simulation engine
│   ├── optimization/
│   │   ├── draft_strategy.py   # Strategy recommendation engine
│   │   └── draft_tracker.py    # Real-time draft state management
│   └── analysis/
│       └── post_draft.py       # Post-draft analysis engine
│
└── data/
    └── cache/                  # Cached simulation results
```

## 🏆 Post-Draft Analysis Features

### Automatically Available After Draft Completion
- **Draft Grade (A-F):** Your performance vs expected value at your draft position
- **Prize Probabilities:** Monte Carlo simulation of win/loss prize chances for all drafters
- **Strategy Analysis:** Detects each drafter's strategy (WIN_FOCUSED, LOSS_FOCUSED, BALANCED)
- **Best/Worst Picks:** Identifies biggest value picks and reaches of the draft
- **Final Standings Prediction:** Predicted finish order with confidence levels
- **Smart Insights:** Personalized recommendations based on your performance

### Visualizations
- **Prize Probability Matrix:** Scatter plot showing each drafter's win/loss chances
- **Performance Metrics:** Tables with your results highlighted
- **Value Analysis:** Complete breakdown of every pick's value

## Key Metrics Explained

- **Expected Wins/Losses:** Mean from simulations
- **P90 Wins/Losses:** 90th percentile (upside potential)
- **Playoff Prob:** Chance of 10+ wins
- **Top Pick Prob:** Chance of 3 or fewer wins
- **Value Score:** Combined metric for draft value
- **Performance Ratio:** Your actual value vs expected value (>1.0 = above expected)

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
- **Post-draft analysis not loading**: Ensure all 32 teams have been drafted
- **Snake draft order incorrect**: Check that `get_current_drafter()` is working properly

## Advanced Features

- Works for any draft position (1-8) with position-specific strategies
- Complete post-draft analysis with Monte Carlo prize probability calculations
- Real-time strategy detection and effectiveness scoring
- Export results for post-draft analysis  
- Adjust strategy weights in real-time
- Override recommendations when needed

## Testing

Run the comprehensive test suite:
```bash
# Test core functionality
python test_system.py

# Test post-draft analysis
python test_post_draft.py

# Test all draft positions
python test_positions.py
```

Good luck with your draft! 🏈