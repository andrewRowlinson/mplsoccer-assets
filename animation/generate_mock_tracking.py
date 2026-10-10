"""Generate mock SkillCorner-format tracking data for mplsoccer's
animation gallery: Rolos Rejects vs Victorious Secret, 60 seconds at
10 fps. Entirely made up no provider or proprietary data.

Run with: uv run python generate_mock_tracking.py
Outputs mock_match.json and mock_tracking_extrapolated.jsonl.
"""
import json

import numpy as np
from scipy.ndimage import gaussian_filter1d

RNG = np.random.default_rng(8)
FPS, SECONDS = 25, 60
N = FPS * SECONDS
START_FRAME = 100
L, W = 105, 68

# ---------------------------------------------------------------- squads
ROLES = {
    'GK': ('Goalkeeper', 'Goalkeeper'), 'RB': ('Right Back', 'Defender'),
    'CB': ('Center Back', 'Defender'), 'LB': ('Left Back', 'Defender'),
    'DM': ('Defensive Midfielder', 'Midfielder'),
    'CM': ('Center Midfielder', 'Midfielder'),
    'RM': ('Right Midfielder', 'Midfielder'),
    'LM': ('Left Midfielder', 'Midfielder'),
    'RW': ('Right Winger', 'Wide Attacker'),
    'LW': ('Left Winger', 'Wide Attacker'), 'ST': ('Striker', 'Attacker'),
}
# (id, number, first, last, role, base_x, base_y) home attack ->
HOME = [  # Rolos Rejects, 4-3-3, defending the left goal
    (101, 1, 'Jordan', 'Pickles', 'GK', -48, 0),
    (102, 66, 'Trent', 'Alexander-Armchair', 'RB', -30, -24),
    (103, 4, 'Virgil', 'van Drains', 'CB', -32, -8),
    (104, 5, 'Rubber', 'Dias', 'CB', -32, 8),
    (105, 3, 'Andy', 'Robbery', 'LB', -30, 24),
    (106, 16, 'Rodri', 'Golightly', 'DM', -15, 0),
    (107, 17, 'Kevin', 'De Brainy', 'CM', -8, -12),
    (108, 10, 'Luka', 'Moderately', 'CM', -8, 12),
    (109, 7, 'Mo', 'Salad', 'RW', 8, -22),
    (110, 9, 'Erling', 'Garland', 'ST', 12, 0),
    (111, 11, 'Vinicius', 'Senior', 'LW', 8, 22),
]
AWAY = [  # Victorious Secret, 4-4-2, defending the right goal
    (201, 1, 'Alisson', 'Wonderland', 'GK', 48, 0),
    (202, 2, 'Kyle', 'Walkies', 'RB', 28, 22),
    (203, 6, 'William', 'Syllabus', 'CB', 30, 8),
    (204, 4, 'John', 'Scones', 'CB', 30, -8),
    (205, 3, 'Alphonso', 'Driveways', 'LB', 28, -22),
    (206, 7, 'Riyad', 'Mayonnaise', 'RM', 12, 20),
    (207, 8, 'Declan', 'Rice-Pudding', 'CM', 10, 7),
    (208, 22, 'Jude', 'Bellyache', 'CM', 10, -7),
    (209, 10, 'Rafael', 'Lionheart', 'LM', 12, -20),
    (210, 9, 'Harry', 'Cane', 'ST', -5, 5),
    (211, 11, 'Lionel', 'Messy', 'ST', -5, -5),
]

# ------------------------------------------------------------- movement
def jitter(n_ent, std):
    """Mean-reverting wobble around zero: smooth, bounded (~std metres)."""
    theta = 0.5 / FPS
    steps = RNG.normal(0, std * np.sqrt(2 * theta), size=(N, n_ent, 2))
    out = np.zeros_like(steps)
    for i in range(1, N):
        out[i] = out[i - 1] * (1 - theta) + steps[i]
    return gaussian_filter1d(out, sigma=0.4 * FPS, axis=0)

home_base = np.array([[p[5], p[6]] for p in HOME], dtype=float)
away_base = np.array([[p[5], p[6]] for p in AWAY], dtype=float)
drift = np.zeros((N, 1, 2))
drift[:, 0, 0] = np.sin(np.linspace(0, 3 * np.pi, N)) * 9
home_xy = home_base + jitter(11, 3.2) + drift
home_xy[:, 0] = home_base[0] + jitter(1, 0.8)[:, 0]   # keeper stays home

# ball: a home passing chain, flight interpolated with a z arc
carriers = [106, 107, 110, 109, 107, 108, 111, 110, 106, 102,
            107, 109, 110, 108, 111, 106, 107, 110]
ids_all = [p[0] for p in HOME + AWAY]
home_ids = [p[0] for p in HOME]
switch = np.linspace(0, N - 1, len(carriers) + 1).astype(int)
ball = np.zeros((N, 3))
for k, pid in enumerate(carriers):
    a, b = switch[k], switch[k + 1]
    col = home_ids.index(pid)
    ball[a:b, :2] = home_xy[a:b, col] + [0.4, 0.2]
    if k:
        prev = home_ids.index(carriers[k - 1])
        frm = home_xy[a - 1, prev] + [0.4, 0.2]
        # flight time from pass length (~16 m/s ball speed)
        dist0 = np.hypot(*(home_xy[a, col] - frm))
        flight = int(np.clip(dist0 / 16 * FPS, 0.3 * FPS, min(1.2 * FPS, b - a)))
        to = home_xy[a + flight - 1, col] + [0.4, 0.2]
        t = np.linspace(0, 1, flight)[:, None]
        ball[a:a + flight, :2] = frm + (to - frm) * t
        dist = np.hypot(*(to - frm))
        ball[a:a + flight, 2] = np.sin(np.pi * t[:, 0]) * min(4, dist / 8)

# away: the 4-4-2 block tracks the (smoothed) ball instead of wandering
ball_s = gaussian_filter1d(ball[:, :2], sigma=1.5 * FPS, axis=0)
shift = (ball_s - ball_s.mean(axis=0)) * [0.6, 0.4]
away_xy = away_base + jitter(11, 2.2) + shift[:, None, :]
away_xy[:, 0] = (away_base[0] + jitter(1, 0.5)[:, 0]
                 + shift[:, None, 1] * [0, 0.5])   # keeper slides across only
players_xy = np.concatenate([home_xy, away_xy], axis=1)
players_xy[..., 0] = players_xy[..., 0].clip(-52, 52)
players_xy[..., 1] = players_xy[..., 1].clip(-33, 33)

# possession: the passing chain's current carrier (all home here)
possession_pid = np.empty(N, dtype=int)
for k, pid in enumerate(carriers):
    possession_pid[switch[k]:switch[k + 1]] = pid
possession_pid[switch[-1]:] = carriers[-1]

# detection flags: mostly true, a few extrapolated spells
detected = np.ones((N, 22), dtype=bool)
for col in RNG.choice(22, 3, replace=False):
    a = RNG.integers(0, N - 6 * FPS)
    detected[a:a + RNG.integers(2 * FPS, 6 * FPS), col] = False

# broadcast camera footprint following the ball
cam = gaussian_filter1d(ball[:, 0], 2 * FPS)

# ---------------------------------------------------------------- write
def player_entry(pid, number, first, last, role, team_id):
    name, group = ROLES[role]
    return {
        'id': pid, 'first_name': first, 'last_name': last,
        'short_name': f'{first[0]}. {last}', 'number': number,
        'team_id': team_id, 'team_player_id': pid, 'trackable_object': pid,
        'player_role': {'id': 0, 'name': name, 'acronym': role,
                        'position_group': group},
        'start_time': '00:00:00', 'end_time': None, 'birthday': None,
        'gender': 'male', 'goal': 0, 'own_goal': 0, 'yellow_card': 0,
        'red_card': 0, 'injured': False,
        'playing_time': {'total': {'minutes_played': 1.0}, 'by_period': []},
    }

match = {
    'id': 9999999, 'status': 'closed', 'date_time': '2026-08-01T10:00:00Z',
    'home_team_score': 2, 'away_team_score': 2, 'home_team_side': ['left_to_right'],
    'home_team': {'id': 901, 'name': 'Rolos Rejects',
                  'short_name': 'Rolos Rejects', 'acronym': 'ROL'},
    'away_team': {'id': 902, 'name': 'Victorious Secret',
                  'short_name': 'Victorious Secret', 'acronym': 'VIC'},
    'home_team_coach': None, 'away_team_coach': None,
    'home_team_kit': {'id': 1, 'team_id': 901, 'season': '2026',
                      'name': 'home', 'jersey_color': '#d7191c',
                      'number_color': '#ffffff'},
    'away_team_kit': {'id': 2, 'team_id': 902, 'season': '2026',
                      'name': 'home', 'jersey_color': '#2c7bb6',
                      'number_color': '#ffffff'},
    'ball': {'trackable_object': 55},
    'competition_edition': {'id': 1, 'competition':
        {'id': 1, 'area': 'XXX', 'name': 'mplsoccer Invitational',
         'gender': 'male', 'age_group': 'adult'},
        'season': {'id': 1, 'start_year': 2026, 'end_year': 2026,
                   'name': '2026'}, 'name': 'mplsoccer Invitational 2026'},
    'competition_round': None, 'referees': [],
    'stadium': {'id': 1, 'name': 'The Crisp Bowl', 'city': 'Sillytown',
                'capacity': 42},
    'pitch_length': L, 'pitch_width': W,
    'match_periods': [{'period': 1, 'name': 'period_1',
                       'start_frame': START_FRAME,
                       'end_frame': START_FRAME + N - 1,
                       'duration_frames': N,
                       'duration_minutes': N / FPS / 60}],
    'players': ([player_entry(*p[:5], 901) for p in HOME]
                + [player_entry(*p[:5], 902) for p in AWAY]),
}
json.dump(match, open('mock_match.json', 'w'))

with open('mock_tracking_extrapolated.jsonl', 'w') as f:
    for i in range(N):
        secs = i / FPS
        frame = {
            'frame': START_FRAME + i,
            'timestamp': f'00:{int(secs // 60):02d}:{secs % 60:05.2f}',
            'period': 1,
            'ball_data': {'x': round(float(ball[i, 0]), 2),
                          'y': round(float(ball[i, 1]), 2),
                          'z': round(float(ball[i, 2]), 2),
                          'is_detected': True},
            'possession': {'player_id': int(possession_pid[i]),
                           'group': 'home team'},
            'image_corners_projection': {
                'x_top_left': round(float(cam[i] - 24), 2),
                'y_top_left': 39.0,
                'x_bottom_left': round(float(cam[i] - 10), 2),
                'y_bottom_left': -16.0,
                'x_bottom_right': round(float(cam[i] + 10), 2),
                'y_bottom_right': -16.0,
                'x_top_right': round(float(cam[i] + 24), 2),
                'y_top_right': 39.0},
            'player_data': [
                {'x': round(float(players_xy[i, k, 0]), 2),
                 'y': round(float(players_xy[i, k, 1]), 2),
                 'player_id': pid,
                 'is_detected': bool(detected[i, k])}
                for k, pid in enumerate(ids_all)],
        }
        f.write(json.dumps(frame) + '\n')
print(f'wrote mock_match.json + mock_tracking_extrapolated.jsonl ({N} frames)')
