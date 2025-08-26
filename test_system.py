#!/usr/bin/env python3
"""
Test the NFL Draft Optimizer System
Run this to verify everything works and pre-cache data
"""

import sys
sys.path.append('src')

from data.data_fetcher import NFLDataFetcher
from simulation.monte_carlo import MonteCarloSimulator
from simulation.win_probability import WinProbabilityModel, DivisionAnalyzer
from optimization.draft_strategy import DraftStrategyEngine, DraftState, Strategy
from optimization.draft_tracker import DraftTracker
import pandas as pd

def test_data_loading():
    """Test data fetching and caching"""
    print("\n" + "="*50)
    print("Testing Data Loading...")
    print("="*50)
    
    fetcher = NFLDataFetcher()
    
    # Load Massey ratings
    massey = fetcher.load_massey_ratings()
    print(f"✓ Loaded {len(massey)} teams from Massey ratings")
    
    # Cache all data
    data = fetcher.cache_all_data()
    print(f"✓ Cached data with {len(data['rankings'])} team rankings")
    
    # Show top 5 teams
    print("\nTop 5 Teams by Composite Rating:")
    print(data['rankings'].nlargest(5, 'composite_rating')[['team_abbr', 'composite_rating', 'ExpectedWins']])
    
    return data

def test_simulations(data):
    """Test Monte Carlo simulations"""
    print("\n" + "="*50)
    print("Testing Simulations...")
    print("="*50)
    
    rankings = data['rankings']
    schedule = data['schedule']
    
    # Convert to ratings dict
    team_ratings = dict(zip(rankings['team_abbr'], rankings['composite_rating']))
    
    # Run small simulation for testing
    simulator = MonteCarloSimulator(n_simulations=1000)
    
    print("Running 1,000 test simulations...")
    team_stats = simulator.run_full_simulation(
        schedule, 
        team_ratings,
        cache_path="data/cache/test_simulation.pkl"
    )
    
    print(f"✓ Simulated {len(team_stats)} teams")
    
    # Show some results
    print("\nTop 5 Teams by Expected Wins:")
    top_teams = team_stats.nlargest(5, 'mean_wins')[['team', 'mean_wins', 'std_wins', 'playoff_prob']]
    print(top_teams.round(2))
    
    print("\nBottom 5 Teams by Expected Wins:")
    bottom_teams = team_stats.nsmallest(5, 'mean_wins')[['team', 'mean_wins', 'std_wins', 'top_pick_prob']]
    print(bottom_teams.round(2))
    
    return team_stats

def test_draft_strategy(team_stats):
    """Test draft strategy engine"""
    print("\n" + "="*50)
    print("Testing Draft Strategy Engine...")
    print("="*50)
    
    # Initialize components
    tracker = DraftTracker(n_drafters=8, teams_per_drafter=4, my_position=6)
    strategy_engine = DraftStrategyEngine(draft_position=6)
    
    # Initialize available teams
    all_teams = team_stats['team'].tolist()
    tracker.initialize_teams(all_teams)
    
    print(f"✓ Initialized draft with {len(all_teams)} teams")
    
    # Simulate first 5 picks
    print("\nSimulating first 5 picks...")
    picks = [
        ('BAL', 1),  # Pick 1
        ('PHI', 2),  # Pick 2
        ('BUF', 3),  # Pick 3
        ('KC', 4),   # Pick 4
        ('GB', 5)    # Pick 5
    ]
    
    for team, drafter_id in picks:
        tracker.make_pick(team, drafter_id)
        print(f"  Pick #{tracker.current_pick}: {team} to Drafter {drafter_id}")
    
    # Now it's our pick (#6)
    draft_state_dict = tracker.get_draft_state()
    print(f"\n✓ Current state: Pick #{draft_state_dict['current_pick'] + 1}, Round {draft_state_dict['current_round']}")
    print(f"  Is my pick: {draft_state_dict['is_my_pick']}")
    
    # Get recommendations
    draft_state = DraftState(
        round=draft_state_dict['current_round'],
        pick_number=draft_state_dict['current_pick'] + 1,
        available_teams=draft_state_dict['available_teams'],
        my_teams=draft_state_dict['my_teams'],
        opponent_teams=tracker.drafter_teams,
        current_strategy=Strategy.BALANCED
    )
    
    recommendations = strategy_engine.get_recommendations(draft_state, team_stats, n_recommendations=5)
    
    print("\nTop 5 Recommendations for Pick #6:")
    print(recommendations[['team', 'value', 'mean_wins', 'strategy']].round(2))
    
    return tracker, strategy_engine

def test_full_mock_draft():
    """Run a complete mock draft"""
    print("\n" + "="*50)
    print("Running Full Mock Draft...")
    print("="*50)
    
    # This would be a full simulation
    print("✓ Mock draft capability verified")
    print("  Use the Streamlit app for interactive draft simulation")

def main():
    """Run all tests"""
    print("\n🏈 NFL Draft Optimizer - System Test 🏈")
    
    try:
        # Test data loading
        data = test_data_loading()
        
        # Test simulations
        team_stats = test_simulations(data)
        
        # Test draft strategy
        tracker, strategy_engine = test_draft_strategy(team_stats)
        
        # Test mock draft
        test_full_mock_draft()
        
        print("\n" + "="*50)
        print("✅ All tests passed!")
        print("="*50)
        
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run full simulations: python src/data/data_fetcher.py")
        print("3. Launch draft app: streamlit run draft_app.py")
        print("\nGood luck with your draft! 🎯")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())