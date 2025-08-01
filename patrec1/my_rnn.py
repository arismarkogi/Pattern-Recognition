import numpy as np

# RNN Model
class SimpleRNN:
    def __init__(self, w_rec, w_x):
        self.w_rec = w_rec  # Recurrent weight
        self.w_x = w_x      # Input weight
    
    def forward(self, x):
        """
        Perform the forward pass through the RNN.
        x: Input binary sequence (numpy array).
        Returns the final output y_T and all states.
        """
        T = len(x)
        s = np.zeros(T)  # States of the RNN
        for t in range(T):
            s[t] = (s[t-1] if t > 0 else 0) * self.w_rec + x[t] * self.w_x
        return s, s[-1]  # All states and final output
    
    def compute_gradients(self, x, t):
        """
        Compute the gradients for w_rec and w_x based on the loss.
        x: Input binary sequence (numpy array).
        t: Target (integer, count of 1s in the sequence).
        Returns the gradients for w_rec and w_x.
        """
        T = len(x)
        s, y_T = self.forward(x)  # Perform forward pass
        error = y_T - t  # Error at the final timestep
        dL_dyT = 2 * error  # Derivative of MSE loss with respect to y_T
        
        # Gradients
        dL_dw_rec = 0
        dL_dw_x = 0
        # Backpropagation Through Time
        for k in range(T-1, -1, -1):
            dL_dw_rec += (s[k-1] if k > 0 else 0) * (self.w_rec ** (T-k-1)) * dL_dyT
            dL_dw_x += x[k] * (self.w_rec ** (T-k-1)) * dL_dyT
        
        return dL_dw_rec, dL_dw_x

    def update_weights(self, gradients, lr):
        """
        Update the weights using gradient descent.
        gradients: Tuple (dL_dw_rec, dL_dw_x).
        lr: Learning rate.
        """
        dL_dw_rec, dL_dw_x = gradients
        self.w_rec -= lr * dL_dw_rec
        self.w_x -= lr * dL_dw_x

# Training the model
# Input sequence
x = np.array([1, 0, 1, 0, 1, 0])  # Binary input sequence
t = np.sum(x)  # Target: Count of 1s in the sequence
T = len(x)

# Initial weights
w_rec_init = [1.0, 0.5, 2,]
w_x_init = [0.0, 0.5, 2]



for w_rec, w_x in zip(w_rec_init, w_x_init):
    print(f"Initial w_rec: {w_rec}, Initial w_x: {w_x}")
    # Initialize the model
    rnn = SimpleRNN(w_rec=w_rec_init, w_x=w_x_init)

    # Training loop
    learning_rate = 0.1
    n_steps = 3  # Number of manual updates
    for step in range(n_steps):
        # Compute gradients
        gradients = rnn.compute_gradients(x, t)
        # Update weights
        rnn.update_weights(gradients, learning_rate)
        # Forward pass to compute new output
        _, y_T = rnn.forward(x)
        # Compute loss
        loss = (t - y_T) ** 2
        print(f"Step {step+1}: Loss = {loss:.4f}, Weights = (w_rec={rnn.w_rec:.4f}, w_x={rnn.w_x:.4f})")

    # Final results
    print("Final weights:", rnn.w_rec, rnn.w_x)
    _, final_output = rnn.forward(x)
    print("Final output:", final_output)
    print()
