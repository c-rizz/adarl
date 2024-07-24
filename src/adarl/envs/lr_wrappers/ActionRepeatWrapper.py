import adarl.utils.dbg.ggLog as ggLog
from adarl.envs.lr_wrappers.LrWrapper import LrWrapper
from adarl.envs.BaseEnv import BaseEnv

class ActionRepeatWrapper(LrWrapper):

    def __init__(self,  env : BaseEnv,
                        action_repeat : int):
        super().__init__(env)
        self._actionToDo = None
        self._action_repeat = action_repeat


    def submitAction(self, action) -> None:
        self._actionToDo = action
        self.env.submitAction(self._actionToDo)

    def performStep(self):
        self._rewardSum = 0
        for i in range(self._action_repeat):
            previousState = self.env.getState()
            self.env.performStep()
            state = self.env.getState()
            self._rewardSum += self.env.computeReward(previousState, state, self._actionToDo, env_conf=self.env.get_configuration())
            self.env.submitAction(self._actionToDo)
        self._previousState = self._lastState
        self._lastState = state

    def performReset(self, options = {}):

        self.env.performReset()

        self._previousState = self.env.getState()
        self._lastState = self.env.getState()

    def computeReward(self, previousState, state, action, env_conf = None, sub_rewards = {}) -> float:
        if not (state is self._lastState and action is self._actionToDo and previousState is self._previousState):
            raise RuntimeError("ActionRepeatWrapper.computeReward is only valid if used for the last executed step. And it looks like you tried using it for something else.")
        return self._rewardSum
