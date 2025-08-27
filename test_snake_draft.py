#!/usr/bin/env python3
"""
Test the snake draft order calculation
"""

def get_current_drafter(pick_number, n_drafters=8):
    """Calculate which drafter should pick based on snake draft order"""
    if pick_number < 1:
        return 1
    
    # Convert to 0-based indexing
    pick_index = pick_number - 1
    
    # Calculate round (0-based) and position in round (0-based)
    round_num = pick_index // n_drafters
    position_in_round = pick_index % n_drafters
    
    # Snake draft: odd rounds (0, 2, 4...) go forward 1-8
    # Even rounds (1, 3, 5...) go backward 8-1
    if round_num % 2 == 0:  # Odd round in 1-based counting
        drafter = position_in_round + 1
    else:  # Even round in 1-based counting
        drafter = n_drafters - position_in_round
    
    return drafter

def test_snake_draft_order():
    """Test the complete snake draft order for 32 picks"""
    print("🏈 Testing Snake Draft Order (8 drafters, 4 rounds)")
    print("=" * 50)
    
    expected_pattern = [
        # Round 1 (picks 1-8): 1, 2, 3, 4, 5, 6, 7, 8
        1, 2, 3, 4, 5, 6, 7, 8,
        # Round 2 (picks 9-16): 8, 7, 6, 5, 4, 3, 2, 1
        8, 7, 6, 5, 4, 3, 2, 1,
        # Round 3 (picks 17-24): 1, 2, 3, 4, 5, 6, 7, 8
        1, 2, 3, 4, 5, 6, 7, 8,
        # Round 4 (picks 25-32): 8, 7, 6, 5, 4, 3, 2, 1
        8, 7, 6, 5, 4, 3, 2, 1
    ]
    
    print("Pick | Expected | Calculated | ✓/✗")
    print("-" * 35)
    
    all_correct = True
    
    for pick in range(1, 33):
        expected = expected_pattern[pick - 1]
        calculated = get_current_drafter(pick)
        correct = "✓" if expected == calculated else "✗"
        
        if expected != calculated:
            all_correct = False
        
        print(f"{pick:2d}   | {expected:8d} | {calculated:10d} | {correct}")
    
    print("-" * 35)
    
    if all_correct:
        print("🎉 ALL TESTS PASSED! Snake draft order is correct.")
    else:
        print("❌ TESTS FAILED! Snake draft order is incorrect.")
    
    # Show the specific pattern mentioned in the issue
    print("\n🔍 Issue Example (end of Round 2 to start of Round 3):")
    test_picks = [15, 16, 17, 18]  # Should be: 3, 2, 1, 8, 7 → Wait that's wrong
    
    print("Pick | Drafter | Round Position")
    print("-" * 30)
    
    for pick in test_picks:
        drafter = get_current_drafter(pick)
        round_num = ((pick - 1) // 8) + 1
        pos_in_round = ((pick - 1) % 8) + 1
        print(f"{pick:2d}   | {drafter:7d} | R{round_num}P{pos_in_round}")
    
    return all_correct

if __name__ == "__main__":
    test_snake_draft_order()