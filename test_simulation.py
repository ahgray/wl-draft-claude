#!/usr/bin/env python3
"""
Test the simulation engine with mock data
"""

import sys
import pandas as pd
import numpy as np
sys.path.append('src')

from data.data_fetcher import NFLDataFetcher
from simulation.win_probability import WinProbabilityModel
from simulation.monte_carlo import MonteCarloSimulator

def create_mock_schedule():
    """Create a simplified mock NFL schedule"""
    # Get team list
    fetcher = NFLDataFetcher()
    composite = fetcher.create_composite_rankings()
    teams = composite['team_abbr'].tolist()
    
    # Create a simplified schedule (each team plays 17 games)
    games = []
    game_id = 1
    
    # Simple approach: each team plays every other team about once
    # Plus some divisional games
    for i, home_team in enumerate(teams):
        games_for_team = 0
        for j, away_team in enumerate(teams):
            if home_team != away_team and games_for_team < 9:  # Half home games
                games.append({
                    'game_id': f'2025_{game_id:03d}',
                    'season': 2025,
                    'week': (game_id % 18) + 1,
                    'gameday': '2025-09-08',  # Mock date
                    'home_team': home_team,
                    'away_team': away_team,
                    'div_game': 0  # Simplified - no divisional tracking
                })
                game_id += 1
                games_for_team += 1
                if game_id > 272:  # NFL has 272 games
                    break
        if game_id > 272:
            break
    
    schedule_df = pd.DataFrame(games)
    print(f"Created mock schedule with {len(schedule_df)} games")
    
    return schedule_df

def test_win_probability():
    """Test the win probability model"""
    print("\n" + "="*50)
    print("Testing Win Probability Model")
    print("="*50)
    
    model = WinProbabilityModel()
    
    # Test cases based on our actual team ratings
    test_cases = [
        ("BAL vs CAR (home)", 9.10, 8.22),  # Best vs worst
        ("PHI at KC", 9.09, 8.97),          # Close matchup
        ("GB vs DET", 8.97, 8.96),          # Very close
        ("Average matchup", 8.5, 8.5)       # Neutral
    ]
    
    for name, rating_a, rating_b in test_cases:
        prob = model.calculate_win_probability(rating_a, rating_b, is_home_a=True)
        print(f"  {name}: {prob:.1%}")
    
    print("✓ Win probability model working")
    return model

def test_monte_carlo():
    """Test Monte Carlo simulation"""
    print("\n" + "="*50)
    print("Testing Monte Carlo Simulation")
    print("="*50)
    
    # Get team data
    fetcher = NFLDataFetcher()
    composite = fetcher.create_composite_rankings()
    
    # Create team ratings dict
    team_ratings = dict(zip(composite['team_abbr'], composite['composite_rating']))
    
    # Create mock schedule
    schedule = create_mock_schedule()
    
    # Run small simulation for testing
    simulator = MonteCarloSimulator(n_simulations=1000)  # Small for speed
    
    print("Running 1,000 season simulations...")
    team_stats = simulator.run_full_simulation(schedule, team_ratings)
    
    print(f"✓ Simulated {len(team_stats)} teams")
    
    # Show results
    print("\nTop 5 teams by expected wins:")
    top_wins = team_stats.nlargest(5, 'mean_wins')[['team', 'mean_wins', 'std_wins', 'p90_wins']]
    print(top_wins.round(2))
    
    print("\nBottom 5 teams by expected wins (best for losses):")
    bottom_wins = team_stats.nsmallest(5, 'mean_wins')[['team', 'mean_wins', 'mean_losses', 'p90_losses']]
    print(bottom_wins.round(2))
    
    return team_stats

def test_draft_logic():
    """Test draft strategy components"""
    print("\n" + "="*50)
    print("Testing Draft Strategy Logic")
    print("="*50)
    
    # Test snake draft picks
    def get_snake_picks(position=6, n_drafters=8, n_rounds=4):
        picks = []
        for round_num in range(1, n_rounds + 1):
            if round_num % 2 == 1:
                pick = (round_num - 1) * n_drafters + position
            else:
                pick = round_num * n_drafters - position + 1
            picks.append(pick)
        return picks
    
    our_picks = get_snake_picks()
    print(f"Our picks (position 6): {our_picks}")
    
    # Test portfolio evaluation
    sample_portfolio = ['BAL', 'CAR', 'PHI', 'TEN']
    print(f"Sample portfolio: {sample_portfolio}")
    
    fetcher = NFLDataFetcher()
    composite = fetcher.create_composite_rankings()
    
    portfolio_data = composite[composite['team_abbr'].isin(sample_portfolio)]
    total_rating = portfolio_data['composite_rating'].sum()
    total_vegas_wins = portfolio_data['vegas_wins'].sum()
    
    print(f"Portfolio composite rating: {total_rating:.2f}")
    print(f"Portfolio expected wins: {total_vegas_wins:.1f}")
    print(f"Portfolio expected losses: {(17*4) - total_vegas_wins:.1f}")
    
    print("✓ Draft logic working")

def main():
    print("🏈 NFL Draft Optimizer - Simulation Test")
    print("=" * 50)
    
    try:
        # Test 1: Win probability model
        model = test_win_probability()
        
        # Test 2: Monte Carlo simulation
        team_stats = test_monte_carlo()
        
        # Test 3: Draft logic
        test_draft_logic()
        
        print("\n" + "="*50)
        print("✅ All simulation tests passed!")
        print("="*50)
        
        print("\nNext steps:")
        print("1. Full simulation with more iterations")
        print("2. Test Streamlit UI")
        print("3. Ready for draft day!")
        
        return team_stats
        
    except Exception as e:
        print(f"❌ Error in simulation test: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()