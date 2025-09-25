#!/usr/bin/env python3
"""
Steve Horses Pro - Fixed with correct API endpoints
Uses the working /v1/north-america/meets pattern from your original code
"""

import os
import sys
import json
import time
import logging
import requests
import numpy as np
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [PRO] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('PRO')

class FixedRacingAPI:
    """Fixed API client using correct endpoints"""
    
    def __init__(self):
        self.session = requests.Session()
        
        # Your working credentials
        self.user = 'WQaKSMwgmG8GnbkHgvRRCT0V'
        self.password = 'McYBoQViXSPvlNcvxQi1Z1py'
        
        # Set up authentication
        import base64
        auth_string = base64.b64encode(f"{self.user}:{self.password}".encode()).decode()
        self.session.headers.update({
            'Authorization': f'Basic {auth_string}',
            'User-Agent': 'Mozilla/5.0'
        })
        
        self.base_url = "https://api.theracingapi.com"
        self.timeout = 10
        
    def get_today_meets(self, date_str=None):
        """Get today's meets using the working endpoint"""
        if not date_str:
            date_str = date.today().isoformat()
        
        # Use the working endpoint pattern from your successful test
        url = f"{self.base_url}/v1/north-america/meets"
        params = {
            'start_date': date_str,
            'end_date': date_str
        }
        
        try:
            logger.info(f"Fetching meets for {date_str}...")
            response = self.session.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                meets = data.get('meets', [])
                logger.info(f"Found {len(meets)} meets")
                return data
            else:
                logger.error(f"API returned status {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch meets: {e}")
            return None
    
    def get_meet_entries(self, meet_id):
        """Get entries for a specific meet"""
        url = f"{self.base_url}/v1/north-america/meets/{meet_id}/entries"
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Entries request failed for {meet_id}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch entries for {meet_id}: {e}")
            return None

class SimpleDataProcessor:
    """Simple data processing using patterns from your working code"""
    
    def __init__(self):
        # Simple model weights from your working versions
        self.weights = {
            'speed': 0.30,
            'class': 0.25, 
            'form': 0.20,
            'trainer': 0.15,
            'jockey': 0.10
        }
    
    def calculate_win_probability(self, horse_data, race_horses):
        """Calculate win probability using simplified model"""
        components = []
        score = 0.0
        
        # Speed rating
        speed = self._get_speed(horse_data)
        if speed:
            field_speeds = [self._get_speed(h) for h in race_horses]
            field_speeds = [s for s in field_speeds if s]
            if field_speeds:
                avg_speed = sum(field_speeds) / len(field_speeds)
                speed_score = (speed - avg_speed + 15) / 30  # Normalize
                speed_score = max(0.0, min(1.0, speed_score))
                score += speed_score * self.weights['speed']
                if speed_score > 0.6:
                    components.append('SPD')
        
        # Class rating
        class_rating = self._get_class(horse_data)
        if class_rating:
            class_score = class_rating / 100.0
            score += class_score * self.weights['class']
            if class_score > 0.6:
                components.append('CLS')
        
        # Form (simplified)
        form = self._get_form(horse_data)
        if form:
            form_score = self._calculate_form_score(form)
            score += form_score * self.weights['form']
            if form_score > 0.6:
                components.append('FRM')
        
        # Trainer/Jockey stats
        trainer_pct = self._get_percentage(horse_data, ['trainer_win_pct', 'trainerWinPct'])
        jockey_pct = self._get_percentage(horse_data, ['jockey_win_pct', 'jockeyWinPct'])
        
        if trainer_pct:
            score += (trainer_pct / 100.0) * self.weights['trainer']
            if trainer_pct > 15:
                components.append('TRN')
        
        if jockey_pct:
            score += (jockey_pct / 100.0) * self.weights['jockey']
            if jockey_pct > 15:
                components.append('JKY')
        
        # Final probability
        win_prob = max(0.01, min(0.95, score))
        
        return win_prob, components
    
    def _get_speed(self, horse_data):
        """Extract speed rating from various possible field names"""
        for field in ['speed', 'spd', 'speed_rating', 'last_speed', 'best_speed']:
            value = horse_data.get(field)
            if value:
                try:
                    return float(value)
                except:
                    continue
        return None
    
    def _get_class(self, horse_data):
        """Extract class rating"""
        for field in ['class', 'class_rating', 'par_class']:
            value = horse_data.get(field)
            if value:
                try:
                    return float(value)
                except:
                    continue
        return None
    
    def _get_form(self, horse_data):
        """Extract form string"""
        for field in ['form', 'recent_form', 'last_runs']:
            value = horse_data.get(field)
            if value:
                return str(value)
        return None
    
    def _get_percentage(self, horse_data, fields):
        """Extract percentage values (trainer/jockey win rates)"""
        for field in fields:
            value = horse_data.get(field)
            if value:
                try:
                    return float(value)
                except:
                    continue
        return None
    
    def _calculate_form_score(self, form_string):
        """Simple form score calculation"""
        if not form_string:
            return 0.5
        
        # Split form into individual race results
        parts = str(form_string).replace('-', ' ').split()[:5]  # Last 5 races
        if not parts:
            return 0.5
        
        score = 0.0
        weights = [0.4, 0.25, 0.15, 0.1, 0.1]  # Recent races weighted more
        
        for i, pos in enumerate(parts):
            if i >= len(weights):
                break
            try:
                position = int(pos)
                if position == 1:
                    score += weights[i]
                elif position <= 3:
                    score += weights[i] * 0.6
                elif position <= 5:
                    score += weights[i] * 0.3
            except:
                continue
        
        return min(1.0, score)

class HTMLGenerator:
    """Generate HTML output similar to your working versions"""
    
    def __init__(self):
        self.prime_threshold = 0.15  # 15%+ edge for prime
        self.action_threshold = 0.08  # 8%+ edge for action
    
    def generate(self, processed_data, output_file):
        """Generate the HTML report"""
        html = self._build_html(processed_data)
        
        # Ensure output directory exists
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"HTML report generated: {output_file}")
    
    def _build_html(self, data):
        """Build the complete HTML report"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Find prime and action plays
        prime_plays = []
        action_plays = []
        
        for track_data in data:
            track_name = track_data['track']
            for race_data in track_data['races']:
                race_num = race_data['race_number']
                for horse in race_data['horses']:
                    edge = horse.get('edge', 0)
                    if edge >= self.prime_threshold:
                        prime_plays.append({
                            'track': track_name,
                            'race': race_num,
                            'horse': horse,
                            'edge': edge
                        })
                    elif edge >= self.action_threshold:
                        action_plays.append({
                            'track': track_name,
                            'race': race_num,
                            'horse': horse,
                            'edge': edge
                        })
        
        # Sort by edge
        prime_plays.sort(key=lambda x: x['edge'], reverse=True)
        action_plays.sort(key=lambda x: x['edge'], reverse=True)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Steve Horses Pro - {timestamp}</title>
    <meta http-equiv="refresh" content="60">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 20px;
        }}
        .timestamp {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
        }}
        .boards {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .board {{
            border-radius: 10px;
            padding: 20px;
            color: white;
            min-height: 200px;
        }}
        .prime-board {{
            background: linear-gradient(135deg, #27ae60, #2ecc71);
        }}
        .action-board {{
            background: linear-gradient(135deg, #f39c12, #e67e22);
        }}
        .board h2 {{
            margin-top: 0;
            margin-bottom: 15px;
        }}
        .play-item {{
            background: rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .play-item:hover {{
            background: rgba(255, 255, 255, 0.3);
        }}
        .race-section {{
            margin-bottom: 30px;
        }}
        .race-header {{
            background: #34495e;
            color: white;
            padding: 15px 20px;
            border-radius: 8px 8px 0 0;
            font-weight: bold;
            font-size: 18px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 0 0 8px 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{
            background: #f8f9fa;
            font-weight: bold;
            color: #2c3e50;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .number {{
            font-weight: bold;
            font-size: 16px;
            color: #e74c3c;
        }}
        .horse-name {{
            font-weight: bold;
            color: #2980b9;
        }}
        .win-prob {{
            font-weight: bold;
            color: #27ae60;
        }}
        .edge-positive {{
            color: #27ae60;
            font-weight: bold;
        }}
        .edge-negative {{
            color: #e74c3c;
        }}
        .components {{
            font-size: 12px;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏇 Steve Horses Pro</h1>
        <div class="timestamp">Last Updated: {timestamp}</div>
        
        <div class="boards">
            <div class="board prime-board">
                <h2>⭐ PRIME PLAYS (≥15% Edge)</h2>
                {self._build_plays_section(prime_plays[:8])}
            </div>
            
            <div class="board action-board">
                <h2>🎯 ACTION PLAYS (≥8% Edge)</h2>
                {self._build_plays_section(action_plays[:8])}
            </div>
        </div>
        
        {self._build_races_section(data)}
    </div>
</body>
</html>"""
        
        return html
    
    def _build_plays_section(self, plays):
        """Build the plays section for prime/action boards"""
        if not plays:
            return "<p>No qualifying plays at this time.</p>"
        
        items = []
        for play in plays:
            horse = play['horse']
            items.append(f"""
                <div class="play-item">
                    <div>
                        <strong>{horse.get('number', '?')} {horse.get('name', 'Unknown')}</strong><br>
                        <small>{play['track']} Race {play['race']}</small>
                    </div>
                    <div style="text-align: right;">
                        <div>Win: {horse.get('win_prob', 0)*100:.1f}%</div>
                        <div>Edge: +{play['edge']*100:.1f}%</div>
                    </div>
                </div>
            """)
        
        return ''.join(items)
    
    def _build_races_section(self, data):
        """Build the races section"""
        sections = []
        
        for track_data in data:
            track_name = track_data['track']
            
            for race_data in track_data['races']:
                race_num = race_data['race_number']
                horses = race_data['horses']
                
                # Sort horses by win probability
                horses.sort(key=lambda x: x.get('win_prob', 0), reverse=True)
                
                rows = []
                for horse in horses:
                    edge = horse.get('edge', 0)
                    edge_class = 'edge-positive' if edge > 0 else 'edge-negative'
                    edge_text = f"+{edge*100:.1f}%" if edge > 0 else f"{edge*100:.1f}%"
                    
                    components = ' '.join(horse.get('components', []))
                    
                    rows.append(f"""
                        <tr>
                            <td class="number">{horse.get('number', '?')}</td>
                            <td class="horse-name">{horse.get('name', 'Unknown')}</td>
                            <td class="win-prob">{horse.get('win_prob', 0)*100:.1f}%</td>
                            <td>{horse.get('market_prob', 0)*100:.1f}%</td>
                            <td class="{edge_class}">{edge_text}</td>
                            <td>{horse.get('fair_odds', 'N/A')}</td>
                            <td>{horse.get('current_odds', 'N/A')}</td>
                            <td class="components">{components}</td>
                        </tr>
                    """)
                
                sections.append(f"""
                    <div class="race-section">
                        <div class="race-header">{track_name} - Race {race_num}</div>
                        <table>
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Horse</th>
                                    <th>Win%</th>
                                    <th>Market%</th>
                                    <th>Edge</th>
                                    <th>Fair Odds</th>
                                    <th>Current</th>
                                    <th>Factors</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join(rows)}
                            </tbody>
                        </table>
                    </div>
                """)
        
        return ''.join(sections)

class SteveHorsesProFixed:
    """Main application using fixed API endpoints"""
    
    def __init__(self):
        logger.info("Starting Steve Horses Pro (Fixed)")
        self.api = FixedRacingAPI()
        self.processor = SimpleDataProcessor()
        self.html_generator = HTMLGenerator()
    
    def run(self):
        """Main execution"""
        try:
            # Get today's date
            today = date.today().isoformat()
            logger.info(f"Processing races for {today}")
            
            # Fetch meets data
            meets_data = self.api.get_today_meets(today)
            if not meets_data:
                logger.error("Failed to fetch meets data")
                return self._generate_error_html()
            
            # Process each meet
            processed_data = []
            
            meets = meets_data.get('meets', [])
            logger.info(f"Processing {len(meets)} meets")
            
            for meet in meets:
                track_name = meet.get('track_name', 'Unknown Track')
                meet_id = meet.get('meet_id')
                
                if not meet_id:
                    logger.warning(f"No meet_id for {track_name}")
                    continue
                
                logger.info(f"Processing {track_name} (ID: {meet_id})")
                
                # Get entries for this meet
                entries_data = self.api.get_meet_entries(meet_id)
                if not entries_data:
                    logger.warning(f"No entries data for {track_name}")
                    continue
                
                races = entries_data.get('races', [])
                logger.info(f"Found {len(races)} races at {track_name}")
                
                track_races = []
                
                for race in races:
                    race_num = race.get('race_number', race.get('number', '?'))
                    runners = race.get('runners', race.get('entries', []))
                    
                    if not runners:
                        continue
                    
                    logger.info(f"Processing {track_name} Race {race_num} ({len(runners)} horses)")
                    
                    # Process each horse
                    processed_horses = []
                    
                    for runner in runners:
                        # Skip scratched horses
                        if runner.get('scratched') or runner.get('withdrawn'):
                            continue
                        
                        # Calculate win probability
                        win_prob, components = self.processor.calculate_win_probability(runner, runners)
                        
                        # Get market probability (from morning line or current odds)
                        market_prob = self._get_market_probability(runner)
                        
                        # Calculate edge
                        edge = win_prob - market_prob if market_prob else 0.0
                        
                        # Calculate fair odds
                        fair_odds = f"{1/win_prob:.1f}" if win_prob > 0.05 else "N/A"
                        
                        # Get current odds
                        current_odds = self._get_current_odds(runner)
                        
                        processed_horses.append({
                            'number': runner.get('program_number', runner.get('number', '?')),
                            'name': runner.get('horse_name', runner.get('name', 'Unknown')),
                            'win_prob': win_prob,
                            'market_prob': market_prob,
                            'edge': edge,
                            'fair_odds': fair_odds,
                            'current_odds': current_odds,
                            'components': components
                        })
                    
                    if processed_horses:
                        track_races.append({
                            'race_number': race_num,
                            'horses': processed_horses
                        })
                
                if track_races:
                    processed_data.append({
                        'track': track_name,
                        'races': track_races
                    })
            
            # Generate HTML output
            if processed_data:
                output_file = Path('outputs') / f"{today}_horses_fixed.html"
                self.html_generator.generate(processed_data, output_file)
                logger.info("Processing complete!")
                return str(output_file)
            else:
                logger.warning("No data to process")
                return self._generate_error_html("No race data found")
        
        except Exception as e:
            logger.error(f"Critical error: {e}")
            return self._generate_error_html(f"System error: {e}")
    
    def _get_market_probability(self, runner):
        """Extract market probability from odds"""
        # Try to get odds from various fields
        odds_fields = ['morning_line', 'ml', 'current_odds', 'odds', 'price']
        
        for field in odds_fields:
            odds_value = runner.get(field)
            if odds_value:
                try:
                    # Handle different odds formats
                    if isinstance(odds_value, str):
                        if '/' in odds_value:  # Fractional odds like "5/2"
                            num, den = odds_value.split('/')
                            decimal_odds = (float(num) / float(den)) + 1
                        else:
                            decimal_odds = float(odds_value)
                    else:
                        decimal_odds = float(odds_value)
                    
                    if decimal_odds > 1:
                        return 1.0 / decimal_odds
                except:
                    continue
        
        return 0.1  # Default 10% if no odds found
    
    def _get_current_odds(self, runner):
        """Get current odds display"""
        odds_fields = ['current_odds', 'morning_line', 'ml', 'odds', 'price']
        
        for field in odds_fields:
            odds_value = runner.get(field)
            if odds_value:
                return str(odds_value)
        
        return "N/A"
    
    def _generate_error_html(self, message="System Error"):
        """Generate error HTML page"""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Steve Horses Pro - Error</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            text-align: center;
            padding: 50px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .error-box {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 40px;
            backdrop-filter: blur(10px);
        }}
        h1 {{ color: #ff6b6b; }}
    </style>
</head>
<body>
    <div class="error-box">
        <h1>⚠️ {message}</h1>
        <p>Please check the logs and try again.</p>
        <p>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>"""
        
        output_file = Path('outputs') / f"{date.today().isoformat()}_error.html"
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_file, 'w') as f:
            f.write(html)
        
        return str(output_file)

if __name__ == '__main__':
    app = SteveHorsesProFixed()
    output = app.run()
    
    # Auto-open on Mac
    if output and sys.platform == 'darwin':
        os.system(f'open "{output}"')