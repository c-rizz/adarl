from lr_gym.envs.BaseEnv import BaseEnv

class LrWrapper(BaseEnv):

    def __init__(self,
                 env : BaseEnv):
        self.env = env

    def __getattr__(self, name):
        '''
        For any attribute not in the wrapper try to call from self.env
        '''
        return getattr(self.env,name)

    def submitAction(self, *args,**kwargs) -> None:
        return self.env.submitAction(*args,**kwargs)

    def reachedTimeout(self):
        return self.env.reachedTimeout()
    
    def checkEpisodeEnded(self, *args,**kwargs):
        return self.env.checkEpisodeEnded(*args,**kwargs)

    def computeReward(self, *args,**kwargs):
        return self.env.computeReward(*args,**kwargs)

    def getObservation(self, *args,**kwargs):
        return self.env.getObservation(*args,**kwargs)

    def getState(self, *args,**kwargs):
        return self.env.getState(*args,**kwargs)

    def initializeEpisode(self, *args,**kwargs):
        return self.env.initializeEpisode(*args,**kwargs)

    def performStep(self, *args,**kwargs):
        return self.env.performStep(*args,**kwargs)

    def performReset(self, options = {}):
        return self.env.performReset(options)

    def getUiRendering(self, *args,**kwargs):
        return self.env.getUiRendering(*args,**kwargs)

    def getInfo(self, *args,**kwargs):
        return self.env.getInfo(*args,**kwargs)

    def getMaxStepsPerEpisode(self, *args,**kwargs):
        return self.env.getMaxStepsPerEpisode(*args,**kwargs)

    def buildSimulation(self, *args,**kwargs):
        return self.env.buildSimulation(*args,**kwargs)

    def _destroySimulation(self, *args,**kwargs):
        return self.env._destroySimulation(*args,**kwargs)

    def getSimTimeFromEpStart(self, *args,**kwargs):
        return self.env.getSimTimeFromEpStart(*args,**kwargs)

    def close(self, *args,**kwargs):
        return self.env.close(*args,**kwargs)

    def seed(self, *args,**kwargs):
        return self.env.seed(*args,**kwargs)

    def get_seed(self, *args,**kwargs):
        return  self.env.get_seed(*args,**kwargs)

    def is_timelimited(self, *args,**kwargs):
        return self.env.is_timelimited(*args,**kwargs)

    def get_configuration(self, *args,**kwargs):
        return self.env.get_configuration(*args,**kwargs)