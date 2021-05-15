class Policy():
    def __init__(self, state_dim, config):
        super(Policy, self).__init__()
        self.config = config
        self.state_space = state_dim
        self.action_dim = config.env.action_space.shape[0]
        
        # TODO: set optimizer