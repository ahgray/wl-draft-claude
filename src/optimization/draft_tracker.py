"""
Real-time Draft Tracker
Manages draft state and provides instant updates
"""

import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

class DraftTracker:
    def __init__(self, 
                 n_drafters: int = 8,
                 teams_per_drafter: int = 4,
                 my_position: int = 6):
        """
        Initialize draft tracker
        
        Args:
            n_drafters: Number of drafters
            teams_per_drafter: Teams each drafter picks
            my_position: Our draft position
        """
        self.n_drafters = n_drafters
        self.teams_per_drafter = teams_per_drafter
        self.my_position = my_position
        self.total_picks = n_drafters * teams_per_drafter
        
        # Initialize draft state
        self.current_pick = 0
        self.draft_board = {}  # pick_number -> (drafter_id, team)
        self.drafter_teams = {i: [] for i in range(1, n_drafters + 1)}
        self.available_teams = set()
        self.pick_history = []
        
        # Calculate snake draft order
        self.draft_order = self._calculate_draft_order()
        
    def _calculate_draft_order(self) -> List[int]:
        """Calculate complete snake draft order"""
        order = []
        for round_num in range(1, self.teams_per_drafter + 1):
            if round_num % 2 == 1:  # Odd rounds
                round_order = list(range(1, self.n_drafters + 1))
            else:  # Even rounds (reverse)
                round_order = list(range(self.n_drafters, 0, -1))
            order.extend(round_order)
        return order
    
    def initialize_teams(self, all_teams: List[str]):
        """Set initial available teams"""
        self.available_teams = set(all_teams)
    
    def make_pick(self, team: str, drafter_id: Optional[int] = None) -> Dict:
        """
        Record a pick in the draft
        
        Args:
            team: Team abbreviation being picked
            drafter_id: ID of drafter (if None, determined by draft order)
        
        Returns:
            Dict with pick details
        """
        if team not in self.available_teams:
            raise ValueError(f"{team} is not available")
        
        self.current_pick += 1
        
        if drafter_id is None:
            drafter_id = self.draft_order[self.current_pick - 1]
        
        # Record pick
        self.draft_board[self.current_pick] = (drafter_id, team)
        self.drafter_teams[drafter_id].append(team)
        self.available_teams.remove(team)
        
        # Add to history
        pick_detail = {
            'pick_number': self.current_pick,
            'round': (self.current_pick - 1) // self.n_drafters + 1,
            'drafter_id': drafter_id,
            'team': team,
            'timestamp': datetime.now().isoformat()
        }
        self.pick_history.append(pick_detail)
        
        return pick_detail
    
    def undo_last_pick(self) -> Optional[Dict]:
        """Undo the last pick made"""
        if self.current_pick == 0:
            return None
        
        # Get last pick details
        last_pick = self.pick_history.pop()
        drafter_id = last_pick['drafter_id']
        team = last_pick['team']
        
        # Restore state
        self.drafter_teams[drafter_id].remove(team)
        self.available_teams.add(team)
        del self.draft_board[self.current_pick]
        self.current_pick -= 1
        
        return last_pick
    
    def get_current_drafter(self) -> int:
        """Get ID of current drafter"""
        if self.current_pick >= self.total_picks:
            return None
        return self.draft_order[self.current_pick]
    
    def get_next_drafter(self) -> Optional[int]:
        """Get ID of next drafter"""
        if self.current_pick + 1 >= self.total_picks:
            return None
        return self.draft_order[self.current_pick + 1]
    
    def is_my_pick(self) -> bool:
        """Check if it's our turn to pick"""
        return self.get_current_drafter() == self.my_position
    
    def get_my_next_pick(self) -> Optional[int]:
        """Get our next pick number"""
        for i in range(self.current_pick, self.total_picks):
            if self.draft_order[i] == self.my_position:
                return i + 1
        return None
    
    def picks_until_my_turn(self) -> int:
        """How many picks until our next turn"""
        next_pick = self.get_my_next_pick()
        if next_pick:
            return next_pick - self.current_pick - 1
        return -1
    
    def get_draft_state(self) -> Dict:
        """Get complete draft state"""
        return {
            'current_pick': self.current_pick,
            'current_round': (self.current_pick) // self.n_drafters + 1,
            'current_drafter': self.get_current_drafter(),
            'my_teams': self.drafter_teams[self.my_position],
            'available_teams': list(self.available_teams),
            'is_my_pick': self.is_my_pick(),
            'picks_until_my_turn': self.picks_until_my_turn(),
            'my_next_pick': self.get_my_next_pick(),
            'draft_complete': self.current_pick >= self.total_picks
        }
    
    def get_drafter_summary(self) -> pd.DataFrame:
        """Get summary of all drafters' picks"""
        summary = []
        for drafter_id, teams in self.drafter_teams.items():
            summary.append({
                'drafter_id': drafter_id,
                'teams': teams,
                'num_teams': len(teams),
                'is_me': drafter_id == self.my_position
            })
        return pd.DataFrame(summary)
    
    def save_draft(self, filepath: str):
        """Save draft state to file"""
        state = {
            'settings': {
                'n_drafters': self.n_drafters,
                'teams_per_drafter': self.teams_per_drafter,
                'my_position': self.my_position
            },
            'current_pick': self.current_pick,
            'draft_board': {str(k): v for k, v in self.draft_board.items()},
            'drafter_teams': self.drafter_teams,
            'available_teams': list(self.available_teams),
            'pick_history': self.pick_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_draft(self, filepath: str):
        """Load draft state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Restore settings
        self.n_drafters = state['settings']['n_drafters']
        self.teams_per_drafter = state['settings']['teams_per_drafter']
        self.my_position = state['settings']['my_position']
        
        # Restore state
        self.current_pick = state['current_pick']
        self.draft_board = {int(k): tuple(v) for k, v in state['draft_board'].items()}
        self.drafter_teams = {int(k): v for k, v in state['drafter_teams'].items()}
        self.available_teams = set(state['available_teams'])
        self.pick_history = state['pick_history']
        
        # Recalculate derived values
        self.total_picks = self.n_drafters * self.teams_per_drafter
        self.draft_order = self._calculate_draft_order()
    
    def simulate_remaining_picks(self, team_rankings: pd.DataFrame) -> Dict:
        """
        Simulate how remaining draft might go
        Useful for 'what-if' analysis
        """
        # Create copy of current state
        sim_available = self.available_teams.copy()
        sim_drafter_teams = {k: v.copy() for k, v in self.drafter_teams.items()}
        
        # Rank available teams
        available_df = team_rankings[team_rankings['team'].isin(sim_available)]
        
        # Simulate remaining picks
        for pick_num in range(self.current_pick + 1, self.total_picks + 1):
            drafter_id = self.draft_order[pick_num - 1]
            
            # Simple strategy: take best available
            if len(available_df) > 0:
                if len(sim_drafter_teams[drafter_id]) < 2:
                    # First half of draft - take best teams
                    best_team = available_df.nlargest(1, 'mean_wins')['team'].values[0]
                else:
                    # Mix it up in later rounds
                    if drafter_id % 2 == 0:
                        best_team = available_df.nlargest(1, 'mean_wins')['team'].values[0]
                    else:
                        best_team = available_df.nsmallest(1, 'mean_wins')['team'].values[0]
                
                sim_drafter_teams[drafter_id].append(best_team)
                available_df = available_df[available_df['team'] != best_team]
        
        return sim_drafter_teams


class DraftAnalyzer:
    """Analyze draft progress and provide insights"""
    
    def __init__(self, tracker: DraftTracker, team_stats: pd.DataFrame):
        self.tracker = tracker
        self.team_stats = team_stats
    
    def analyze_drafter_strategies(self) -> pd.DataFrame:
        """Analyze apparent strategy of each drafter"""
        analyses = []
        
        for drafter_id, teams in self.tracker.drafter_teams.items():
            if not teams:
                continue
            
            team_data = self.team_stats[self.team_stats['team'].isin(teams)]
            
            analysis = {
                'drafter_id': drafter_id,
                'teams': teams,
                'avg_wins': team_data['mean_wins'].mean(),
                'total_wins': team_data['mean_wins'].sum(),
                'total_losses': (17 * len(teams)) - team_data['mean_wins'].sum(),
                'win_std': team_data['mean_wins'].std(),
                'strategy': self._infer_strategy(team_data)
            }
            analyses.append(analysis)
        
        return pd.DataFrame(analyses)
    
    def _infer_strategy(self, team_data: pd.DataFrame) -> str:
        """Infer strategy from picks"""
        if len(team_data) < 2:
            return "UNKNOWN"
        
        avg_wins = team_data['mean_wins'].mean()
        std_wins = team_data['mean_wins'].std()
        
        if avg_wins > 9:
            return "WIN_FOCUSED"
        elif avg_wins < 7:
            return "LOSS_FOCUSED"
        elif std_wins > 2:
            return "HIGH_VARIANCE"
        else:
            return "BALANCED"
    
    def calculate_portfolio_strength(self, teams: List[str]) -> Dict:
        """Calculate strength metrics for a portfolio"""
        if not teams:
            return {}
        
        team_data = self.team_stats[self.team_stats['team'].isin(teams)]
        
        return {
            'total_expected_wins': team_data['mean_wins'].sum(),
            'total_expected_losses': (17 * len(teams)) - team_data['mean_wins'].sum(),
            'win_upside': team_data['p90_wins'].sum(),
            'loss_upside': team_data['p90_losses'].sum(),
            'consistency': 1 / (team_data['std_wins'].mean() + 1),
            'playoff_teams': (team_data['playoff_prob'] > 0.5).sum(),
            'tank_teams': (team_data['top_pick_prob'] > 0.3).sum()
        }
    
    def identify_value_picks(self) -> pd.DataFrame:
        """Identify value picks still available"""
        available = list(self.tracker.available_teams)
        available_stats = self.team_stats[self.team_stats['team'].isin(available)].copy()
        
        # Calculate value score
        pick_position = self.tracker.current_pick + 1
        expected_rank = pick_position / 32 * len(available_stats)
        
        available_stats['actual_rank'] = available_stats['mean_wins'].rank(ascending=False)
        available_stats['value_score'] = expected_rank - available_stats['actual_rank']
        
        # Higher value score = better than expected to be available
        value_picks = available_stats.nlargest(5, 'value_score')[
            ['team', 'mean_wins', 'actual_rank', 'value_score']
        ]
        
        return value_picks