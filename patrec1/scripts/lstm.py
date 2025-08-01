import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import math
from parser import parser
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from itertools import product
from plot_confusion_matrix import plot_confusion_matrix


output_dim = 10  # number of digits
# TODO: YOUR CODE HERE
# Play with variations of these hyper-parameters and report results
rnn_size = 64
num_layers = 2
bidirectional = True
#dropout = 0.5
#batch_size = 32
patience = 3
epochs = 30
#lr = 1e-3
#weight_decay = 0.01


class EarlyStopping(object):
    def __init__(self, patience, mode="min", base=None):
        self.best = base
        self.patience = patience
        self.patience_left = patience
        self.mode = mode

    def stop(self, value: float) -> bool:
        # TODO: YOUR CODE HERE
        # Decrease patience if the metric has not improved
        # Stop when patience reaches zero
        if not self.has_improved(value):
            self.patience_left -= 1
            if self.patience_left == 0:
                return True
        else:
            self.patience_left = self.patience
        return False

    def has_improved(self, value: float) -> bool:
        # TODO: YOUR CODE HERE
        # Check if the metric has improved
        if self.best is None:
            self.best = value
            return True
        if self.mode == "min" and value <= self.best:
            self.best = value
            return True
        elif self.mode == "max" and value >= self.best:
            self.best = value
            return True
        return False

class FrameLevelDataset(Dataset):
    def __init__(self, feats, labels):
        """
        feats: Python list of numpy arrays that contain the sequence features.
               Each element of this list is a numpy array of shape seq_length x feature_dimension
        labels: Python list that contains the label for each sequence (each label must be an integer)
        """
        # TODO: YOUR CODE HERE
        self.lengths = [len(i) for i in feats]

        self.feats = self.zero_pad_and_stack(feats)
        if isinstance(labels, (list, tuple)):
            self.labels = np.array(labels).astype("int64")

        

    def zero_pad_and_stack(self, x: np.ndarray) -> np.ndarray:
        """
        This function performs zero padding on a list of features and forms them into a numpy 3D array
        returns
            padded: a 3D numpy array of shape num_sequences x max_sequence_length x feature_dimension
        """
        
        # TODO: YOUR CODE HERE
        # --------------- Insert your code here ---------------- #

        max_seq_len = max(len(seq) for seq in x)
        feature_dim = x[0].shape[1] if x else 0
        padded = np.zeros((len(x), max_seq_len, feature_dim))
        for i, seq in enumerate(x):
            padded[i, :len(seq)] = seq
        return padded

    def __getitem__(self, item):
        feats = self.feats[item].astype(np.float32)  # Ensure features are float32
        labels = self.labels[item]
        lengths = self.lengths[item]
        return feats, labels, lengths

    def __len__(self):
        return len(self.feats)


class BasicLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        rnn_size,
        output_dim,
        num_layers,
        bidirectional=False,
        dropout=0.0,
    ):
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size

        # TODO: YOUR CODE HERE
        # --------------- Insert your code here ---------------- #
        # Initialize the LSTM, Dropout, Output layers

        self.lstm = nn.LSTM(
            input_dim,
            self.feature_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout,
        )
        
        # Dropout Layer
        self.dropout = nn.Dropout(dropout)
        
        # Output Layer
        self.fc = nn.Linear(self.feature_size, output_dim)

    def forward(self, x, lengths):
        """
        x : 3D numpy array of dimension N x L x D
            N: batch index
            L: sequence index
            D: feature index

        lengths: N x 1
        """

        # TODO: YOUR CODE HERE
        # --------------- Insert your code here ---------------- #

        # You must have all of the outputs of the LSTM, but you need only the last one (that does not exceed the sequence length)
        # To get it use the last_timestep method
        # Then pass it through the remaining network
        # Forward pass through the LSTM layer
        x = x.to(torch.float32)

        lstm_out, _ = self.lstm(x)  # Shape: (batch_size, sequence_length, hidden_size * num_directions)
        
        # Obtain the last relevant output for each sequence using `last_timestep`
        last_outputs = self.last_timestep(lstm_out, lengths, self.bidirectional)
        
        # Apply dropout to the last relevant outputs
        last_outputs = self.dropout(last_outputs)
        
        # Pass through the final fully connected layer
        return self.fc(last_outputs)

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
        Returns the last output of the LSTM taking into account the zero padding
        """
        # TODO: READ THIS CODE AND UNDERSTAND WHAT IT DOES
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        # TODO: READ THIS CODE AND UNDERSTAND WHAT IT DOES
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # TODO: READ THIS CODE AND UNDERSTAND WHAT IT DOES
        # Index of the last output for each sequence.
        idx = (
            (lengths - 1)
            .view(-1, 1)
            .expand(outputs.size(0), outputs.size(2))
            .unsqueeze(1)
        )
        return outputs.gather(1, idx).squeeze()
    

import torch.nn.utils.rnn as rnn_utils

class PackedLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        rnn_size,
        output_dim,
        num_layers,
        bidirectional=False,
        dropout=0.0,
    ):
        super(PackedLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size

        # Initialize the LSTM, Dropout, Output layers
        self.lstm = nn.LSTM(
            input_dim,
            rnn_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.feature_size, output_dim)

    def forward(self, x, lengths):
        """
        x : 3D numpy array of dimension N x L x D
            N: batch index
            L: sequence index
            D: feature index

        lengths: N x 1
        """
        # Sort the batch by sequence length in descending order
        lengths, perm_idx = lengths.sort(0, descending=True)
        x = x[perm_idx]

        # Pack the padded batch of sequences
        packed_input = rnn_utils.pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Forward pass through LSTM
        packed_output, (h_n, c_n) = self.lstm(packed_input)

        # Unpack the sequences
        lstm_out, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)

        # Restore the original order
        _, unperm_idx = perm_idx.sort(0)
        lstm_out = lstm_out[unperm_idx]

        # Obtain the last relevant output for each sequence
        last_outputs = self.last_timestep(lstm_out, lengths, self.bidirectional)

        # Apply dropout and pass through the final fully connected layer
        last_outputs = self.dropout(last_outputs)
        return self.fc(last_outputs)

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
        Returns the last output of the LSTM taking into account the zero padding
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            return torch.cat((last_forward, last_backward), dim=-1)
        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        idx = (
            (lengths - 1)
            .view(-1, 1)
            .expand(outputs.size(0), outputs.size(2))
            .unsqueeze(1)
        )
        return outputs.gather(1, idx).squeeze()



def create_dataloaders(batch_size):
    X, X_test, y, y_test, spk, spk_test = parser("recordings", n_mfcc=13)

    X_train, X_val, y_train, y_val, spk_train, spk_val = train_test_split(
        X, y, spk, test_size=0.2, stratify=y
    )

    trainset = FrameLevelDataset(X_train, y_train)
    validset = FrameLevelDataset(X_val, y_val)
    testset = FrameLevelDataset(X_test, y_test)
    # TODO: YOUR CODE HERE
    # Initialize the training, val and test dataloaders (torch.utils.data.DataLoader)
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def training_loop(model, train_dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    num_batches = 0
    for num_batch, batch in enumerate(train_dataloader):
        features, labels, lengths = batch
        # TODO: YOUR CODE HERE
        # zero grads in the optimizer
        # run forward pass
        optimizer.zero_grad()  # Clear gradients
        outputs = model(features, lengths)  # Forward pass
        # calculate loss
        loss = criterion(outputs, labels)

        # TODO: YOUR CODE HERE
        # Run backward pass

        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        
        running_loss += loss.item()
        num_batches += 1
    train_loss = running_loss / num_batches
    return train_loss


def evaluation_loop(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    num_batches = 0
    y_pred = torch.empty(0, dtype=torch.int64)
    y_true = torch.empty(0, dtype=torch.int64)
    with torch.no_grad():
        for num_batch, batch in enumerate(dataloader):
            features, labels, lengths = batch

            # TODO: YOUR CODE HERE
            # Run forward pass
            logits = model(features, lengths)  # Forward pass
            loss = criterion(logits, labels)  # Compute loss
            # calculate loss
            loss = criterion(logits, labels)
            running_loss += loss.item()
            # Predict
            outputs = logits.argmax(dim=1)  # Calculate the argmax of logits
            y_pred = torch.cat((y_pred, outputs))
            y_true = torch.cat((y_true, labels))
            num_batches += 1
    valid_loss = running_loss / num_batches
    return valid_loss, y_pred, y_true


def train(train_dataloader, val_dataloader, criterion):
    # TODO: YOUR CODE HERE
    input_dim = train_dataloader.dataset.feats.shape[-1]
    model = BasicLSTM(
        input_dim,
        rnn_size,
        output_dim,
        num_layers,
        bidirectional=bidirectional,
        dropout=dropout,
    )
    # TODO: YOUR CODE HERE
    # Initialize AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    early_stopping = EarlyStopping(patience, mode="min")
    for epoch in range(epochs):
        training_loss = training_loop(model, train_dataloader, optimizer, criterion)
        valid_loss, y_pred, y_true = evaluation_loop(model, val_dataloader, criterion)

        # TODO: Calculate and print accuracy score
        valid_accuracy = accuracy_score(y_true, y_pred)
        print(f"Epoch {epoch}: train loss = {training_loss:.4f}, validation loss = {valid_loss:.4f}, validation accuracy {valid_accuracy:.4f}")
        
        # Check for improvement and save the best model
        if early_stopping.has_improved(valid_loss):
            torch.save(model.state_dict(), "best_model.pth")


        if early_stopping.stop(valid_loss):
            print("early stopping...")
            break
    
    print("Loading the best model from checkpoint...")
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    return model

def train_packed(train_dataloader, val_dataloader, criterion):
    # TODO: YOUR CODE HERE
    input_dim = train_dataloader.dataset.feats.shape[-1]
    model = PackedLSTM(
        input_dim,
        rnn_size,
        output_dim,
        num_layers,
        bidirectional=bidirectional,
        dropout=dropout,
    )
    # TODO: YOUR CODE HERE
    # Initialize AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    early_stopping = EarlyStopping(patience, mode="min")
    for epoch in range(epochs):
        training_loss = training_loop(model, train_dataloader, optimizer, criterion)
        valid_loss, y_pred, y_true = evaluation_loop(model, val_dataloader, criterion)

        # TODO: Calculate and print accuracy score
        valid_accuracy = accuracy_score(y_true, y_pred)
        print(f"Epoch {epoch}: train loss = {training_loss:.4f}, validation loss = {valid_loss:.4f}, validation accuracy {valid_accuracy:.4f}")
        
        # Check for improvement and save the best model
        if early_stopping.has_improved(valid_loss):
            torch.save(model.state_dict(), "best_model.pth")


        if early_stopping.stop(valid_loss):
            print("early stopping...")
            break
    
    print("Loading the best model from checkpoint...")
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    return model

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Per Epoch")
    plt.legend()
    plt.grid()
    plt.savefig(f"images/losses.png")
    #plt.show()


def grid_search(criterion):
    """
    Performs grid search over specified hyperparameters for the BasicLSTM model.
    
    Args:
        criterion (nn.Module): Loss function.
        
    Returns:
        best_model (nn.Module): The best model found during grid search.
        best_params (tuple): The hyperparameters of the best model.
        best_valid_acc (float): Validation accuracy of the best model.
        all_results (list): List of all results with hyperparameters and corresponding accuracies.
        best_train_losses (list): Training losses per epoch for the best model.
        best_val_losses (list): Validation losses per epoch for the best model.
    """
    # Define hyperparameter grid
    batch_sizes = [16, 32]
    learning_rates = [1e-3, 1e-4]
    dropouts = [0.3, 0.5]
    weight_decays = [0.01, 0.001]
    
    best_model = None
    best_valid_acc = 0.0  # Initialize best validation accuracy
    best_params = None
    all_results = []
    
    # Iterate over all combinations of hyperparameters
    for batch_size, lr, dropout, weight_decay in product(batch_sizes, learning_rates, dropouts, weight_decays):
        print(f"\nTraining with batch_size={batch_size}, lr={lr}, dropout={dropout}, weight_decay={weight_decay}")
        
        # Create dataloaders with the current batch_size
        train_dataloader, val_dataloader, test_dataloader = create_dataloaders(batch_size)
        
        # Initialize the model
        input_dim = train_dataloader.dataset.feats.shape[-1]
        model = BasicLSTM(
            input_dim=input_dim,
            rnn_size=rnn_size,
            output_dim=output_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        
        # Initialize the optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=patience, mode="min")
        
        # Lists to store loss per epoch
        train_losses, val_losses = [], []
        
        for epoch in range(epochs):
            train_loss = training_loop(model, train_dataloader, optimizer, criterion)
            val_loss, val_pred, val_true = evaluation_loop(model, val_dataloader, criterion)
            val_acc = accuracy_score(val_true, val_pred)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
            
            # Check for improvement
            if early_stopping.has_improved(val_loss):
                torch.save(model.state_dict(), "best_model.pth")
                print("Validation loss improved. Saving model.")
            
            if early_stopping.stop(val_loss):
                print("Early stopping triggered.")
                break
        
        # Load the best model for this hyperparameter combination
        model.load_state_dict(torch.load("best_model.pth", weights_only=True))
        
        # Evaluate on the test set
        test_loss, test_pred, test_true = evaluation_loop(model, test_dataloader, criterion)
        test_acc = accuracy_score(test_true, test_pred)
        
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        
        # Append results
        all_results.append({
            'batch_size': batch_size,
            'learning_rate': lr,
            'dropout': dropout,
            'weight_decay': weight_decay,
            'validation_accuracy': val_acc,
            'test_accuracy': test_acc
        })
        
        # Update the best model if current model has higher validation accuracy
        if val_acc > best_valid_acc:
            best_valid_acc = val_acc
            best_model = model
            best_params = {
                'batch_size': batch_size,
                'learning_rate': lr,
                'dropout': dropout,
                'weight_decay': weight_decay
            }
            best_train_losses = train_losses
            best_val_losses = val_losses
    
    # After grid search, evaluate the best model on validation and test sets
    print("\n--- Grid Search Completed ---")
    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Validation Accuracy: {best_valid_acc:.4f}")
    
    # Recreate dataloaders with the best batch size
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(best_params['batch_size'])
    
    # Plot training and validation loss for the best model
    plot_loss(best_train_losses, best_val_losses)
    
    labels = list(set(val_true))

    # Evaluate on the validation set
    val_loss, val_pred, val_true = evaluation_loop(best_model, val_dataloader, criterion)
    accuracy_val = accuracy_score(val_true, val_pred)
    print(f"Final Validation Accuracy: {accuracy_val:.4f}")
    confusion_matrix_val =  confusion_matrix(val_true, val_pred)
    plot_confusion_matrix(confusion_matrix_val, labels, normalize= False, title='Confusion Matrix - Validation Set (Best Model) LSTM', cmap=plt.cm.Blues)
    
    # Evaluate on the test set
    test_loss, test_pred, test_true = evaluation_loop(best_model, test_dataloader, criterion)
    accuracy_test = accuracy_score(test_true, test_pred)
    print(f"Final Test Accuracy: {accuracy_test:.4f}")
    confusion_matrix_test =  confusion_matrix(test_true, test_pred)
    plot_confusion_matrix(confusion_matrix_test, labels, normalize= False, title='Confusion Matrix - Test Set (Best Model) LSTM', cmap=plt.cm.Blues)
    
    return best_model, best_params, best_valid_acc, all_results, best_train_losses, best_val_losses

#train_dataloader, val_dataloader, test_dataloader = create_dataloaders(batch_size)
# TODO: YOUR CODE HERE
# Choose an appropriate loss function
criterion = nn.CrossEntropyLoss()

"""
model = train(train_dataloader, val_dataloader, criterion)
model = train_packed(train_dataloader, val_dataloader, criterion)

test_loss, test_pred, test_true = evaluation_loop(model, test_dataloader, criterion)


# TODO: YOUR CODE HERE
# print test loss and test accuracy

test_loss, test_pred, test_true = evaluation_loop(model, test_dataloader, criterion)
print(f"Test loss: {test_loss}, Test accuracy: {accuracy_score(test_true, test_pred)}")
"""
# Perform the grid search
best_model, best_params, best_valid_acc, results, train_losses, val_losses = grid_search(
    criterion, 
)

# Evaluate on the test set
#test_loss, test_pred, test_true = evaluation_loop(best_model, test_dataloader, criterion)
#test_acc = accuracy_score(test_true, test_pred)