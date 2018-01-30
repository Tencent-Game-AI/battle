from battle.arena_sc2 import register, BattleType 

register(
  id='sc2env',
  type=BattleType.env,
  entry_point='pysc2.env.sc2_env:SC2Env',
  map_name='Simple64',
  screen_size_px=(64, 64),
  minimap_size_px=(64, 64),
  agent_race=None,
  bot_race=None,
  difficulty=None,
  step_mul=8,
  game_steps_per_episode=0,
  visualize=True,
)

from .single_env import SingleEnv
