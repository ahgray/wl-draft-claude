# NFL Team Draft Decision Support System

## Project Overview
A decision support system for a fantasy football league where participants draft entire NFL teams (not individual players) to optimize for either most total wins or most total losses across their 4-team portfolio.

## League Rules
- **Participants**: 8 drafters
- **Draft Format**: Snake order
- **Teams per Participant**: 4 NFL teams each
- **Prize Pools**: 
  - One for most total wins
  - One for most total losses
- **Timing**: Draft occurs before 2025 NFL season starts

## Technical Architecture

### Core Components

#### 1. Data Pipeline (`src/data/`)
- **data_fetcher.py**: Centralized data acquisition
  - NFL schedule (272 games with home/away)
  - Preseason team ratings (Massey, ESPN, etc.)
  - Historical performance data (2022-2024)
  - Cache all data locally before draft

#### 2. Simulation Engine (`src/simulation/`)
- **win_probability.py**: Convert ratings to game probabilities
  - Logistic regression model with calibrated k-factor
  - Home field advantage (2.5-3 points)
  - Rest differential adjustments
  - Weather factors for outdoor stadiums
  
- **monte_carlo.py**: Season simulation (20,000+ runs)
  - Vectorized numpy operations for performance
  - Output: Win/loss distributions for all 32 teams
  - Calculate EV, P10, P50, P90, variance

#### 3. Portfolio Optimization (`src/optimization/`)
- **portfolio.py**: Track and evaluate team combinations
  - Marginal value calculations
  - Correlation penalties for shared opponents
  - Head-to-head cannibalization tracking
  
- **strategy.py**: Draft strategy implementation
  - Dual-objective optimization (wins vs losses)
  - Strategy slider (e.g., 70% losses, 30% wins)
  - Round-based strategy adjustments

#### 4. User Interface (`src/ui/`)
- **streamlit_app.py**: Main draft interface
  - Pre-draft: Tier boards (W1-W5, L1-L5)
  - Live draft: Real-time recommendations
  - Portfolio tracking and analytics

### Data Schema

```python
# Schedule DataFrame
schedule_df = {
    'week': int,
    'home_team': str,
    'away_team': str,
    'date': datetime,
    'is_divisional': bool,
    'is_primetime': bool
}

# Ratings Dictionary
ratings = {
    'team_abbr': {
        'massey': float,      # Massey Ratings
        'elo': float,         # FiveThirtyEight Elo
        'srs': float,         # Pro Football Reference SRS
        'power_rank': int,    # ESPN/NFL power rankings
        'composite': float    # Weighted average
    }
}

# Simulation Results Array
# Shape: (32 teams, 20000 simulations)
simulation_results = np.array([...])

# Portfolio State
portfolio = {
    'drafter_id': int,
    'teams': List[str],
    'expected_wins': float,
    'expected_losses': float,
    'p90_wins': float,
    'p10_losses': float,
    'internal_matchups': int
}
```

### Key Algorithms

#### Win Probability Model
```python
def calculate_win_probability(team_a_rating, team_b_rating, is_home_a=True):
    """
    Calculate win probability using calibrated logistic model
    
    Factors:
    - Rating differential
    - Home field advantage (2.65 points historically)
    - K-factor calibrated from historical data (~0.04)
    """
    home_advantage = 2.65 if is_home_a else -2.65
    rating_diff = team_a_rating - team_b_rating + home_advantage
    k_factor = 0.04  # Calibrated from historical variance
    
    return 1 / (1 + np.exp(-k_factor * rating_diff))
```

#### Marginal Value Calculation
```python
def calculate_marginal_value(team, current_portfolio, strategy_weights):
    """
    Calculate the marginal contribution of adding a team to portfolio
    
    Components:
    1. Expected value change
    2. Variance impact
    3. Correlation penalty
    4. Tail outcome improvements (P90/P10)
    """
    # Base EV contribution
    marginal_ev_wins = team.ev_wins
    marginal_ev_losses = team.ev_losses
    
    # Correlation penalties
    correlation_penalty = 0
    for owned_team in current_portfolio:
        # Penalize head-to-head matchups
        h2h_games = count_matchups(team, owned_team)
        correlation_penalty += h2h_games * 0.5
        
        # Penalize shared opponents
        shared_opponents = count_shared_opponents(team, owned_team)
        correlation_penalty += shared_opponents * 0.1
    
    # Calculate final value based on strategy
    win_value = marginal_ev_wins - correlation_penalty
    loss_value = marginal_ev_losses - correlation_penalty
    
    return (strategy_weights['wins'] * win_value + 
            strategy_weights['losses'] * loss_value)
```

### Implementation Phases

#### Phase 1: Data Infrastructure (Week 1)
- [ ] Set up data fetching from nfl_data_py
- [ ] Implement Massey Ratings scraper
- [ ] Create local SQLite cache
- [ ] Build data validation and cleaning

#### Phase 2: Simulation Engine (Week 2)
- [ ] Implement win probability model
- [ ] Calibrate k-factor using 2024 season
- [ ] Build Monte Carlo simulation
- [ ] Optimize with numpy vectorization
- [ ] Validate against historical outcomes

#### Phase 3: Portfolio Logic (Week 3)
- [ ] Create portfolio tracking system
- [ ] Implement marginal value calculations
- [ ] Build correlation matrix
- [ ] Add strategy slider functionality

#### Phase 4: User Interface (Week 4)
- [ ] Design Streamlit layout
- [ ] Create pre-draft tier boards
- [ ] Build live draft tracker
- [ ] Add recommendation engine
- [ ] Implement export functionality

### Performance Requirements
- Simulation: <5 seconds for 20,000 season runs
- Marginal value calculation: <100ms per team
- UI refresh: <500ms after each pick
- Memory usage: <500MB for all cached data

### Testing Strategy
- Unit tests for win probability calculations
- Integration tests for data pipeline
- Backtesting against 2023-2024 seasons
- Monte Carlo validation (convergence tests)
- UI/UX testing with mock drafts

### Deployment
- Local deployment via Streamlit
- All data cached before draft day
- No external API calls during draft
- Backup data sources configured
- Export results to CSV/JSON

## File Structure
```
nfl-draft-optimizer/
├── claude.md
├── requirements.txt
├── .env.example
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_fetcher.py
│   │   ├── cache_manager.py
│   │   └── validators.py
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── win_probability.py
│   │   ├── monte_carlo.py
│   │   └── season_simulator.py
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── portfolio.py
│   │   ├── strategy.py
│   │   └── correlation.py
│   └── ui/
│       ├── __init__.py
│       ├── streamlit_app.py
│       ├── components.py
│       └── visualizations.py
├── tests/
│   ├── test_simulation.py
│   ├── test_portfolio.py
│   └── test_data.py
├── data/
│   ├── cache/
│   ├── raw/
│   └── processed/
└── notebooks/
    ├── calibration.ipynb
    ├── backtest_2024.ipynb
    └── strategy_analysis.ipynb
```

## Dependencies
```python
# Core
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
numba>=0.57.0  # For JIT compilation

# Data
nfl_data_py>=0.3.0
beautifulsoup4>=4.12.0
requests>=2.31.0
requests_cache>=1.1.0

# Database
sqlite3  # Built-in
sqlalchemy>=2.0.0

# UI
streamlit>=1.28.0
plotly>=5.17.0
altair>=5.1.0

# Development
pytest>=7.4.0
black>=23.0.0
mypy>=1.5.0
python-dotenv>=1.0.0
```

## Environment Variables
```bash
# .env file
NFL_DATA_CACHE_DIR=./data/cache
SIMULATION_RUNS=20000
HOME_FIELD_ADVANTAGE=2.65
K_FACTOR=0.04
DEBUG_MODE=false
```

## Next Steps
1. Initialize project with this structure
2. Implement data fetching pipeline
3. Calibrate win probability model with historical data
4. Build and validate Monte Carlo simulation
5. Create portfolio optimization logic
6. Develop Streamlit interface
7. Run backtests and mock drafts
8. Deploy for draft day

## Key Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Fetch and cache all data
python src/data/data_fetcher.py --cache-all

# Run simulation calibration
python src/simulation/calibrate.py --season 2024

# Launch draft interface
streamlit run src/ui/streamlit_app.py

# Run tests
pytest tests/ -v

# Run mock draft
python src/optimization/mock_draft.py --players 8 --strategy mixed
```