#!/usr/bin/env python3
"""
Initial setup script for NFL Team Draft Decision Support System
Run this first to create project structure and fetch initial data
"""

import os
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime
import subprocess

def create_project_structure():
    """Create the complete project directory structure"""
    
    base_dirs = [
        "src",
        "src/data",
        "src/simulation",
        "src/optimization", 
        "src/ui",
        "tests",
        "data",
        "data/cache",
        "data/raw",
        "data/processed",
        "notebooks",
        "logs"
    ]
    
    print("📁 Creating project structure...")
    for dir_path in base_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files for Python packages
        if dir_path.startswith("src"):
            init_file = Path(dir_path) / "__init__.py"
            if not init_file.exists():
                init_file.write_text('"""Package initialization"""')
    
    print("✅ Project structure created")

def create_env_file():
    """Create .env file with default configuration"""
    
    env_content = """# NFL Draft Optimizer Configuration
# Created: {timestamp}

# Data Settings
NFL_DATA_CACHE_DIR=./data/cache
DATA_REFRESH_HOURS=24
USE_CACHED_DATA=true

# Simulation Parameters
SIMULATION_RUNS=20000
HOME_FIELD_ADVANTAGE=2.65
K_FACTOR=0.04
USE_WEATHER_ADJUSTMENTS=true
USE_REST_ADJUSTMENTS=true

# Optimization Settings
DEFAULT_WIN_WEIGHT=0.5
DEFAULT_LOSS_WEIGHT=0.5
CORRELATION_PENALTY_H2H=0.5
CORRELATION_PENALTY_SHARED=0.1

# UI Settings
STREAMLIT_PORT=8501
REFRESH_RATE_MS=500
SHOW_DEBUG_INFO=false

# API Keys (add your own)
# SPORTRADAR_API_KEY=
# ESPN_API_KEY=

# Performance
USE_MULTIPROCESSING=true
MAX_WORKERS=4
CACHE_EXPIRY_DAYS=7

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/nfl_draft.log
""".format(timestamp=datetime.now().isoformat())
    
    env_file = Path(".env")
    if not env_file.exists():
        env_file.write_text(env_content)
        print("✅ Created .env configuration file")
    else:
        print("⚠️  .env file already exists, skipping")

def create_database():
    """Initialize SQLite database for caching"""
    
    db_path = Path("data/cache/nfl_data.db")
    
    print("🗄️  Initializing SQLite database...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.executescript("""
    -- Team ratings table
    CREATE TABLE IF NOT EXISTS team_ratings (
        team_abbr TEXT PRIMARY KEY,
        team_name TEXT,
        massey_rating REAL,
        elo_rating REAL,
        srs_rating REAL,
        power_rank INTEGER,
        composite_rating REAL,
        last_updated TIMESTAMP,
        season INTEGER
    );
    
    -- Schedule table
    CREATE TABLE IF NOT EXISTS schedule (
        game_id TEXT PRIMARY KEY,
        season INTEGER,
        week INTEGER,
        game_date DATE,
        home_team TEXT,
        away_team TEXT,
        is_divisional BOOLEAN,
        is_primetime BOOLEAN,
        stadium TEXT,
        is_dome BOOLEAN,
        FOREIGN KEY (home_team) REFERENCES team_ratings(team_abbr),
        FOREIGN KEY (away_team) REFERENCES team_ratings(team_abbr)
    );
    
    -- Historical results for calibration
    CREATE TABLE IF NOT EXISTS historical_results (
        game_id TEXT PRIMARY KEY,
        season INTEGER,
        week INTEGER,
        home_team TEXT,
        away_team TEXT,
        home_score INTEGER,
        away_score INTEGER,
        home_won BOOLEAN
    );
    
    -- Simulation results cache
    CREATE TABLE IF NOT EXISTS simulation_cache (
        cache_key TEXT PRIMARY KEY,
        season INTEGER,
        simulation_date TIMESTAMP,
        num_simulations INTEGER,
        results_json TEXT,
        parameters_json TEXT
    );
    
    -- Draft history
    CREATE TABLE IF NOT EXISTS draft_history (
        draft_id TEXT,
        pick_number INTEGER,
        round INTEGER,
        drafter_id INTEGER,
        team_abbr TEXT,
        timestamp TIMESTAMP,
        strategy_weights TEXT,
        marginal_value REAL,
        PRIMARY KEY (draft_id, pick_number)
    );
    
    CREATE INDEX IF NOT EXISTS idx_schedule_season ON schedule(season);
    CREATE INDEX IF NOT EXISTS idx_schedule_week ON schedule(week);
    CREATE INDEX IF NOT EXISTS idx_historical_season ON historical_results(season);
    """)
    
    conn.commit()
    conn.close()
    
    print("✅ Database initialized")

def create_sample_config():
    """Create sample configuration file for strategies"""
    
    config = {
        "draft_config": {
            "num_drafters": 8,
            "teams_per_drafter": 4,
            "snake_draft": True,
            "season": 2025
        },
        "strategies": {
            "aggressive_wins": {
                "win_weight": 1.0,
                "loss_weight": 0.0,
                "risk_tolerance": "high"
            },
            "aggressive_losses": {
                "win_weight": 0.0,
                "loss_weight": 1.0,
                "risk_tolerance": "high"
            },
            "balanced": {
                "win_weight": 0.5,
                "loss_weight": 0.5,
                "risk_tolerance": "medium"
            },
            "chaos_seeker": {
                "win_weight": 0.3,
                "loss_weight": 0.7,
                "risk_tolerance": "very_high",
                "prefer_high_variance": True
            }
        },
        "rating_weights": {
            "massey": 0.35,
            "elo": 0.30,
            "srs": 0.20,
            "power_rank": 0.15
        },
        "simulation_params": {
            "default_runs": 20000,
            "convergence_threshold": 0.001,
            "use_parallel": True
        }
    }
    
    config_file = Path("config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Created sample configuration file")

def install_dependencies():
    """Install required Python packages"""
    
    print("📦 Installing Python dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print("Please run: pip install -r requirements.txt manually")

def create_starter_notebook():
    """Create a Jupyter notebook for initial exploration"""
    
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# NFL Team Draft Optimizer - Data Exploration\n",
                          "This notebook helps you explore the data and test the simulation engine"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import nfl_data_py as nfl\n",
                    "from datetime import datetime\n",
                    "import plotly.express as px\n",
                    "import plotly.graph_objects as go\n",
                    "\n",
                    "print('Libraries loaded successfully!')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Fetch 2024 Season Data for Calibration"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Fetch 2024 season results for calibration\n",
                    "games_2024 = nfl.import_schedules([2024])\n",
                    "print(f'Loaded {len(games_2024)} games from 2024 season')\n",
                    "games_2024.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Calculate Team Standings"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Calculate wins/losses by team\n",
                    "completed_games = games_2024[games_2024['home_score'].notna()].copy()\n",
                    "\n",
                    "# Calculate home wins\n",
                    "completed_games['home_won'] = completed_games['home_score'] > completed_games['away_score']\n",
                    "\n",
                    "# Aggregate wins by team\n",
                    "home_wins = completed_games.groupby('home_team')['home_won'].sum()\n",
                    "away_wins = completed_games.groupby('away_team').apply(lambda x: (~x['home_won']).sum())\n",
                    "\n",
                    "total_wins = (home_wins.add(away_wins, fill_value=0)).sort_values(ascending=False)\n",
                    "print('2024 Season Final Standings (Wins):')\n",
                    "print(total_wins.head(10))"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.9.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    notebook_path = Path("notebooks/exploration.ipynb")
    with open(notebook_path, 'w') as f:
        json.dump(notebook_content, f, indent=2)
    
    print("✅ Created starter notebook")

def main():
    """Run all setup tasks"""
    
    print("\n🏈 NFL Team Draft Optimizer - Initial Setup 🏈\n")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    
    # Create project structure
    create_project_structure()
    
    # Create configuration files
    create_env_file()
    create_sample_config()
    
    # Initialize database
    create_database()
    
    # Create starter notebook
    create_starter_notebook()
    
    # Install dependencies (optional - comment out if you want to do manually)
    response = input("\n📦 Install Python dependencies now? (y/n): ")
    if response.lower() == 'y':
        install_dependencies()
    
    print("\n" + "=" * 50)
    print("✅ Setup complete! Next steps:")
    print("1. Review and update .env file with any API keys")
    print("2. Run 'jupyter lab' to explore the starter notebook")
    print("3. Use Claude Code to implement the components")
    print("4. Run 'streamlit run src/ui/streamlit_app.py' when ready")
    print("\n🎯 Happy drafting!")

if __name__ == "__main__":
    main()