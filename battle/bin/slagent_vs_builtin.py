from battle.arena_sc2 import SC2WrapperAgent, make, ids

import battle.arena_sc2.arenas
import battle.arena_sc2.agents
import battle.arena_sc2.envs

from absl import flags
from absl import app

def run_agent_vs_builtin(unused_argv):
  print("all register id=", ids())
  sc2_battle = make('AgentVSBuiltIn')
  sc2_battle.register_agent(
      SC2WrapperAgent(make('SLAgent', model_path='/home/zakzheng/job/git/battle_ai/battle/battle/model/agent.model-2540000')))
  sc2_battle.register_env(make('sc2env'))
  sc2_battle.play_game()
  print(sc2_battle.stat)

if __name__ == '__main__':
  app.run(run_agent_vs_builtin)

