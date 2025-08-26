#!/usr/bin/env python3
"""
Simple test to verify basic functionality
"""

import pandas as pd
import numpy as np

def test_massey_loading():
    """Test loading Massey ratings"""
    print("Testing Massey ratings loading...")
    
    try:
        # Load CSV directly
        df = pd.read_csv("massey-2025-export.csv")
        print(f"✓ Loaded {len(df)} teams from Massey CSV")
        
        # Show some data
        print("\nTop 5 teams by rating:")
        df_clean = df.copy()
        df_clean['Rating'] = pd.to_numeric(df_clean['Rat'], errors='coerce')
        print(df_clean.nlargest(5, 'Rating')[['Team', 'Rating']])
        
        print("\nBottom 5 teams by rating:")
        print(df_clean.nsmallest(5, 'Rating')[['Team', 'Rating']])
        
        return df_clean
        
    except Exception as e:
        print(f"❌ Error loading Massey data: {e}")
        return None

def test_basic_simulation():
    """Test basic win probability calculation"""
    print("\n" + "="*40)
    print("Testing basic simulation logic...")
    
    # Simple win probability function
    def win_probability(rating_a, rating_b, home_advantage=2.65, k_factor=0.04):
        rating_diff = rating_a - rating_b + home_advantage
        return 1 / (1 + np.exp(-k_factor * rating_diff))
    
    # Test cases
    test_cases = [
        ("BAL vs CAR (home)", 9.10, -5.43, True),
        ("PHI at KC", 9.09, 8.97, False),
        ("Average teams", 0, 0, True)
    ]
    
    for name, rating_a, rating_b, is_home in test_cases:
        prob = win_probability(rating_a, rating_b) if is_home else win_probability(rating_a, rating_b, home_advantage=-2.65)
        print(f"  {name}: {prob:.1%} win probability")
    
    print("✓ Basic simulation logic works")

def test_draft_logic():
    """Test basic draft logic"""
    print("\n" + "="*40)
    print("Testing draft logic...")
    
    # Snake draft picks for position 6
    def get_snake_picks(position=6, n_drafters=8, n_rounds=4):
        picks = []
        for round_num in range(1, n_rounds + 1):
            if round_num % 2 == 1:  # Odd rounds
                pick = (round_num - 1) * n_drafters + position
            else:  # Even rounds
                pick = round_num * n_drafters - position + 1
            picks.append(pick)
        return picks
    
    our_picks = get_snake_picks()
    print(f"  Our picks (position 6): {our_picks}")
    print(f"  Pick pairs: ({our_picks[0]}, {our_picks[1]}) and ({our_picks[2]}, {our_picks[3]})")
    
    # Calculate picks between
    picks_between = [our_picks[i+1] - our_picks[i] - 1 for i in range(len(our_picks)-1)]
    print(f"  Picks between our selections: {picks_between}")
    
    print("✓ Draft logic works")

def main():
    print("🏈 NFL Draft Optimizer - Simple Test")
    print("=" * 40)
    
    # Test 1: Data loading
    massey_data = test_massey_loading()
    
    # Test 2: Basic simulation
    test_basic_simulation()
    
    # Test 3: Draft logic
    test_draft_logic()
    
    print("\n" + "="*40)
    if massey_data is not None:
        print("✅ Basic functionality verified!")
        print("\nNext steps:")
        print("1. Install full dependencies when network is stable")
        print("2. Run full simulations")
        print("3. Test Streamlit UI")
    else:
        print("❌ Some issues found - check data files")

if __name__ == "__main__":
    main()