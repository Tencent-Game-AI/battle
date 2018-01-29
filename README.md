## Install
```shell
pip install -e . 
```

## Dependencies
. pysc2 
  Starcraft II gamecore

## Register 
### Register Agent
. you need add code to the file battle/arena_sc2/agents/__init__.py
e.g. register RandomAgent
```shell
register(
  id='RandomAgent',
  type=BattleType.agent,
  entry_point='pysc2.agents.random_agent:RandomAgent',
)
```

### Register Environment 
. you need add code to the file battle/arena_sc2/envs/__init__.py
e.g. Register SC2ENV 
```shell
register(
  id='sc2env',
  type=BattleType.env,
  entry_point='pysc2.env.sc2_env:SC2Env',
  map_name='Simple64',
  screen_size_px=(84, 84),
  minimap_size_px=(64, 64),
  agent_race=None,
  bot_race=None,
  difficulty=None,
  step_mul=8,
  game_steps_per_episode=0,
  visualize=True,
)
```

### Register Arena
. you need add code to the file battle/arena_sc2/envs/__init__.py
e.g. Register AgentVSbuilt-in
```shell
register(
  id='AgentVSBuiltIn',
  type=BattleType.arena,
  entry_point='battle.arena_sc2.arenas.sc2_builtin:SC2AgentVSBuildIn',
)
```

### use
. run Agent VS built-in, e.g..
```shell
  sc2_battle = make('AgentVSBuiltIn')
  sc2_battle.register_agent(SC2WrapperAgent(make('RandomAgent')))
  sc2_battle.register_env(make('sc2env'))
  sc2_battle.play_game()
```

