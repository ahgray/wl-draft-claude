#!/usr/bin/env python3
"""
Test snake draft picks for all draft positions
"""

def get_snake_picks(position, n_drafters=8, n_rounds=4):
    """Calculate snake draft pick numbers for any position"""
    picks = []
    for round_num in range(1, n_rounds + 1):
        if round_num % 2 == 1:  # Odd rounds
            pick = (round_num - 1) * n_drafters + position
        else:  # Even rounds
            pick = round_num * n_drafters - position + 1
        picks.append(pick)
    return picks

def test_all_positions():
    """Test and display picks for all draft positions"""
    print("🏈 Snake Draft Picks by Position (8 drafters, 4 rounds)")
    print("=" * 60)
    print()
    
    # Calculate pick gaps for analysis
    def analyze_picks(picks):
        """Analyze pick patterns"""
        gaps = []
        for i in range(len(picks)-1):
            gaps.append(picks[i+1] - picks[i] - 1)
        return gaps
    
    # Test each position
    for position in range(1, 9):
        picks = get_snake_picks(position)
        gaps = analyze_picks(picks)
        
        print(f"Position #{position}:")
        print(f"  Picks: {', '.join(map(str, picks))}")
        print(f"  Gaps between picks: {', '.join(map(str, gaps))}")
        
        # Strategic analysis
        if position <= 2:
            strategy = "Elite teams available, can show hand early"
        elif position <= 4:
            strategy = "Strong tier 1 options, balanced approach"
        elif position <= 6:
            strategy = "Maximum flexibility, pivot potential"
        else:
            strategy = "Value plays, snake advantage in round 2"
        
        print(f"  Strategy: {strategy}")
        print()
    
    # Show snake advantage analysis
    print("Snake Draft Advantages:")
    print("-" * 40)
    print("Position 1: Gets first pick but longest wait (14 picks)")
    print("Position 8: Latest first pick but back-to-back picks")
    print("Positions 4-5: Most consistent gaps between picks")
    print()
    
    # Show round transitions
    print("Round Transition Pick Pairs:")
    print("-" * 40)
    for position in range(1, 9):
        picks = get_snake_picks(position)
        print(f"Position {position}: R1→R2: {picks[0]}→{picks[1]} (gap: {picks[1]-picks[0]-1})")

def test_specific_position(position):
    """Detailed test for a specific position"""
    print(f"\n🎯 Detailed Analysis for Position #{position}")
    print("=" * 50)
    
    picks = get_snake_picks(position, 8, 4)
    
    print(f"Your picks: {picks}")
    print()
    
    # Analyze each round
    for i, pick in enumerate(picks, 1):
        round_num = i
        pick_in_round = ((pick - 1) % 8) + 1
        
        print(f"Round {round_num}:")
        print(f"  Overall pick: #{pick}")
        print(f"  Position in round: {pick_in_round}/8")
        
        if i < len(picks):
            wait_time = picks[i] - pick - 1
            print(f"  Picks until next selection: {wait_time}")
        print()
    
    # Strategic recommendations
    print("Strategic Recommendations:")
    if position <= 3:
        print("- Target elite teams in R1 (BAL, PHI, BUF)")
        print("- Can afford to show strategy early")
        print("- Watch for value in R3 after long wait")
    elif position <= 6:
        print("- Flexible approach in R1")
        print("- Quick R2 pick allows strategy pivot")
        print("- Good position for blocking in R3")
    else:
        print("- Best value teams likely available in R1")
        print("- Back-to-back picks in R1/R2 = strategy flexibility")
        print("- Consider chaos strategy with variance")

if __name__ == "__main__":
    # Test all positions
    test_all_positions()
    
    # Detailed test for position 6 (as example)
    test_specific_position(6)
    
    print("\n✅ All position tests complete!")