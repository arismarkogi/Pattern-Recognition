import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set frequency and interval
f = 40  # frequency in Hz
T = 1 / f  # period

# Create time intervals
num_sequences = 20  # number of sequences for the dataset
sequence_length = 10  # number of points in each sequence
t_intervals = np.linspace(0, T / 3, sequence_length)

# Generate training data
sine_sequences_train = []
cosine_sequences_train = []

for _ in range(num_sequences):
    phase_shift = np.random.uniform(0, T)  # add variability in the phase shift
    sine_seq = np.sin(2 * np.pi * f * (t_intervals + phase_shift))
    cosine_seq = np.cos(2 * np.pi * f * (t_intervals + phase_shift))
    
    sine_sequences_train.append(sine_seq)
    cosine_sequences_train.append(cosine_seq)

# Convert to PyTorch tensors
sine_sequences_train = torch.tensor(sine_sequences_train, dtype=torch.float32).view(num_sequences, sequence_length, 1)
cosine_sequences_train = torch.tensor(cosine_sequences_train, dtype=torch.float32).view(num_sequences, sequence_length, 1)

# Create validation set similarly
sine_sequences_val = []
cosine_sequences_val = []

for _ in range(int(num_sequences * 0.25)):  # 20% of data for validation
    phase_shift = np.random.uniform(0, T)
    sine_seq = np.sin(2 * np.pi * f * (t_intervals + phase_shift))
    cosine_seq = np.cos(2 * np.pi * f * (t_intervals + phase_shift))
    
    sine_sequences_val.append(sine_seq)
    cosine_sequences_val.append(cosine_seq)

sine_sequences_val = torch.tensor(sine_sequences_val, dtype=torch.float32).view(-1, sequence_length, 1)
cosine_sequences_val = torch.tensor(cosine_sequences_val, dtype=torch.float32).view(-1, sequence_length, 1)

# Model definition
class SineToCosinePredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=1, rnn_type="LSTM"):
        super(SineToCosinePredictor, self).__init__()
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

# Initialize LSTM model, loss function, and optimizer
model = SineToCosinePredictor(rnn_type="LSTM")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop with validation
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(sine_sequences_train)
    loss = criterion(output, cosine_sequences_train)
    loss.backward()
    optimizer.step()
    
    # Validation
    if (epoch + 1) % 50 == 0:
        model.eval()
        with torch.no_grad():
            val_output = model(sine_sequences_val)
            val_loss = criterion(val_output, cosine_sequences_val)
        
        print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')



model.eval()
with torch.no_grad():
    # Predict on the validation set
    predicted_sequences = model(sine_sequences_val).view(-1, sequence_length).numpy()
    actual_sequences = cosine_sequences_val.view(-1, sequence_length).numpy()

    # Plot the actual vs. predicted cosines
    plt.figure(figsize=(12, 8))
    for i in range(int(num_sequences * 0.25)):
        plt.subplot(int(num_sequences * 0.25), 1, i + 1)
        plt.plot(actual_sequences[i], label='Actual Cosine', color='blue')
        plt.plot(predicted_sequences[i], label='Predicted Cosine', color='red', linestyle='--')
        plt.title(f'Sequence {i + 1}')
        plt.xlabel('Timestep')
        plt.ylabel('Cosine Value')
        plt.legend()
        plt.tight_layout()

    plt.show()
    plt.savefig('images/cosine_lstm.png')


    # Initialize LSTM model, loss function, and optimizer
model = SineToCosinePredictor(rnn_type="GRU")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop with validation
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(sine_sequences_train)
    loss = criterion(output, cosine_sequences_train)
    loss.backward()
    optimizer.step()
    
    # Validation
    if (epoch + 1) % 50 == 0:
        model.eval()
        with torch.no_grad():
            val_output = model(sine_sequences_val)
            val_loss = criterion(val_output, cosine_sequences_val)
        
        print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')



model.eval()
with torch.no_grad():
    # Predict on the validation set
    predicted_sequences = model(sine_sequences_val).view(-1, sequence_length).numpy()
    actual_sequences = cosine_sequences_val.view(-1, sequence_length).numpy()

    # Plot the actual vs. predicted cosines
    plt.figure(figsize=(12, 8))
    for i in range(int(num_sequences * 0.25)):
        plt.subplot(int(num_sequences * 0.25), 1, i + 1)
        plt.plot(actual_sequences[i], label='Actual Cosine', color='blue')
        plt.plot(predicted_sequences[i], label='Predicted Cosine', color='red', linestyle='--')
        plt.title(f'Sequence {i + 1}')
        plt.xlabel('Timestep')
        plt.ylabel('Cosine Value')
        plt.legend()
        plt.tight_layout()

    plt.show()
    plt.savefig('images/cosine_gru.png')