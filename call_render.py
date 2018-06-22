import gym
gym.undo_logger_setup()


class CallRender(gym.Wrapper):
    """Call Env.render before every step."""

    def _step(self, action):
        self.env.render()
        return self.env.step(action)
