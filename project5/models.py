import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        return -1
    
    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        converged = False
        while not converged:
            converged = True
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    self.w.update(x, nn.as_scalar(y))
                    converged = False


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # Hidden layer sizes: between 100 and 500.
        # Batch size: between 1 and 128. For Q2 and Q3, we require that total size of the dataset be evenly divisible by the batch size.
        # Learning rate: between 0.0001 and 0.01.
        # Number of hidden layers: between 1 and 3(It’s especially important that you start small here
        self.hidden_layer_size = 100
        self.batch_size = 50
        self.learning_rate = 0.001

        self.W1 = nn.Parameter(1,self.hidden_layer_size)
        self.b1 = nn.Parameter(1,self.hidden_layer_size)

        self.W2 = nn.Parameter(self.hidden_layer_size,1)
        self.b2 = nn.Parameter(1,1)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        # output vector f(x) would be given by the function: f(x) = relu(x * W1 + b1) * W2 + b2
        output_vector1 = nn.AddBias(nn.Linear(x, self.W1), self.b1)
        relu = nn.ReLU(output_vector1)
        output_vector2 = nn.AddBias(nn.Linear(relu, self.W2), self.b2)
        return output_vector2
        

    def get_loss(self, x, y):
        """
        Computes the current_loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a current_loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        for x, y in dataset.iterate_forever(self.batch_size):
            current_loss = self.get_loss(x, y)
            if nn.as_scalar(current_loss) > 0.01:
                gradients = nn.gradients(current_loss, [self.W1, self.W2, self.b1, self.b2])
                self.W1.update(gradients[0], -self.learning_rate)
                self.W2.update(gradients[1], -self.learning_rate)
                self.b1.update(gradients[2], -self.learning_rate)
                self.b2.update(gradients[3], -self.learning_rate)
            else:
                return

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_layer_size = 100
        self.batch_size = 100
        self.learning_rate = 0.1

        self.W1 = nn.Parameter(784,self.hidden_layer_size)
        self.b1 = nn.Parameter(1,self.hidden_layer_size)

        self.W2 = nn.Parameter(self.hidden_layer_size,10) #10 is the number of classes in the output
        self.b2 = nn.Parameter(1,10)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        #same with function run in previous question
        output_vector1 = nn.AddBias(nn.Linear(x, self.W1), self.b1)
        relu = nn.ReLU(output_vector1)
        output_vector2 = nn.AddBias(nn.Linear(relu, self.W2), self.b2)
        return output_vector2


    def get_loss(self, x, y):
        """
        Computes the current_loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a current_loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        #similar with Q2
        for x, y in dataset.iterate_forever(self.batch_size):
            current_loss = self.get_loss(x, y)
            if dataset.get_validation_accuracy() < 0.975:
                gradients = nn.gradients(current_loss, [self.W1, self.W2, self.b1, self.b2])
                self.W1.update(gradients[0], -self.learning_rate)
                self.W2.update(gradients[1], -self.learning_rate)
                self.b1.update(gradients[2], -self.learning_rate)
                self.b2.update(gradients[3], -self.learning_rate)
            else:
                print(f"Validation accuracy : {dataset.get_validation_accuracy()}")
                return

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_layer_size = 100
        self.batch_size = 50
        self.learning_rate = 0.1
        
        self.W_x = nn.Parameter(self.num_chars, self.hidden_layer_size)

        self.W_hidden = nn.Parameter(self.hidden_layer_size, self.hidden_layer_size)

        self.W_output = nn.Parameter(self.hidden_layer_size, len(self.languages))

        self.b_hidden = nn.Parameter(1, self.hidden_layer_size)
        self.b_output = nn.Parameter(1, len(self.languages))

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x self.hidden_layer_size), for your
        choice of self.hidden_layer_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        h = nn.Linear(xs[0], self.W_x)
        relu = nn.ReLU(nn.AddBias(h, self.b_hidden))

        for x in xs[1:]:
            z = nn.Add(nn.Linear(x, self.W_x), nn.Linear(relu, self.W_hidden))
            relu = nn.ReLU(nn.AddBias(z, self.b_hidden))

        output_vector = nn.AddBias(nn.Linear(relu, self.W_output), self.b_output)
        return output_vector

    def get_loss(self, xs, y):
        """
        Computes the current_loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a current_loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        for x, y in dataset.iterate_forever(self.batch_size):
            current_loss = self.get_loss(x, y)
            if dataset.get_validation_accuracy() < 0.89:
                gradients = nn.gradients(current_loss, [self.W_x, self.W_hidden, self.W_output, self.b_hidden, self.b_output])
                self.W_x.update(gradients[0], -self.learning_rate)
                self.W_hidden.update(gradients[1], -self.learning_rate)
                self.W_output.update(gradients[2], -self.learning_rate)
                self.b_hidden.update(gradients[3], -self.learning_rate)
                self.b_output.update(gradients[4], -self.learning_rate)
            else:
                print(f"Validation accuracy : {dataset.get_validation_accuracy()}")
                return
