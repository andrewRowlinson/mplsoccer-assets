# Animation gallery assets

Mock tracking data for mplsoccer's animation examples, in SkillCorner
file format so it loads with `fast-forward-football`. Entirely made up
(Rolos Rejects vs Victorious Secret). No provider or proprietary data.

## Files

- `mock_match.json`: match metadata (teams, players, pitch size)
- `mock_tracking_extrapolated.jsonl`: 60 s of tracking at 10 fps
- `generate_mock_tracking.py`: regenerates both (edit + rerun freely)

## Format notes (learned the hard way)

- fast-forward drops frames whose `possession.group` is null (dead
  ball), so the generator emits a possession carrier on every frame.
- The tracking `player_id` must equal the match.json players' `id`
  (not `trackable_object`).
- `match_periods` start/end frames must cover the jsonl frame range.
