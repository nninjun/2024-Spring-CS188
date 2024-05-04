import nn

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.1
        self.numTrainingGames = 1000
        self.batch_size = 32

        self.fc1_weights = nn.Parameter(state_dim, 128)
        self.fc1_bias = nn.Parameter(1, 128)
        self.fc2_weights = nn.Parameter(128, 64)
        self.fc2_bias = nn.Parameter(1, 64)
        self.fc3_weights = nn.Parameter(64, action_dim)
        self.fc3_bias = nn.Parameter(1, action_dim)
        self.parameters = [self.fc1_weights, self.fc1_bias, self.fc2_weights, self.fc2_bias, self.fc3_weights, self.fc3_bias]

    def set_weights(self, layers):
        self.parameters = []
        for i in range(len(layers)):
            self.parameters.append(layers[i])

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        Q_predictions = self.run(states)
        return nn.SquareLoss(Q_predictions, Q_target)

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        "*** YOUR CODE HERE ***"
        if not isinstance(states, nn.Constant):
            states = nn.Constant(states)
        
        x = states
        x = nn.ReLU(nn.AddBias(nn.Linear(x, self.fc1_weights), self.fc1_bias))
        x = nn.ReLU(nn.AddBias(nn.Linear(x, self.fc2_weights), self.fc2_bias))
        q_values = nn.AddBias(nn.Linear(x, self.fc3_weights), self.fc3_bias)
        return q_values
    
    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"
        loss = self.get_loss(states, Q_target)
        gradients = nn.gradients(loss, self.parameters)

        for param, grad in zip(self.parameters, gradients):
            param.update(grad, -self.learning_rate)