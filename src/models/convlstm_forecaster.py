import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple

def create_sequences(data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a 1D array into overlapping sequences of length `window_size`.
    Returns X (samples, time_steps) and y (samples,).
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

class SimpleConvLSTM(torch.nn.Module):
    """
    A lightweight 1D Convolutional LSTM architecture.
    Applies a 1D convolution across the time dimension to extract local patterns,
    then feeds the sequence into an LSTM.
    """
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, kernel_size=3):
        super(SimpleConvLSTM, self).__init__()
        
        # 1D Conv layer expects input of shape (batch, channels, seq_len)
        self.conv1d = nn.Conv1d(
            in_channels=input_size, 
            out_channels=hidden_size, 
            kernel_size=kernel_size, 
            padding='same'
        )
        self.relu = nn.ReLU()
        
        # LSTM expects input of shape (batch, seq_len, input_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Final fully connected layer to output a single prediction
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # Input x shape: (batch, seq_len, features)
        # Permute for Conv1D: (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        
        c_out = self.conv1d(x)
        c_out = self.relu(c_out)
        
        # Permute back for LSTM: (batch, seq_len, features)
        c_out = c_out.permute(0, 2, 1)
        
        # Extract last hidden state
        lstm_out, _ = self.lstm(c_out)
        last_out = lstm_out[:, -1, :]
        
        out = self.fc(last_out)
        return out

def train_convlstm_model(X_train: np.ndarray, y_train: np.ndarray, epochs: int = 10, batch_size: int = 64) -> SimpleConvLSTM:
    """
    Trains the ConvLSTM model.
    Assumes X_train has shape (samples, time_steps) and y_train has shape (samples,).
    """
    # Reshape X to (samples, time_steps, features)
    X_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = SimpleConvLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    return model

def predict_convlstm_model(model: SimpleConvLSTM, initial_sequence: np.ndarray, future_steps: int) -> np.ndarray:
    """
    Auto-regressive inference: uses the model to predict the next step, appends it 
    to the sequence, and repeats for `future_steps`.
    `initial_sequence` must be of shape (window_size,).
    """
    model.eval()
    predictions = []
    
    # current_seq shape: (1, window_size, 1)
    current_seq = torch.tensor(initial_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    
    with torch.no_grad():
        for _ in range(future_steps):
            pred = model(current_seq)
            pred_val = pred.item()
            predictions.append(pred_val)
            
            # Slide window forward
            # remove first element, append new prediction
            pred_tensor = torch.tensor([[[pred_val]]], dtype=torch.float32)
            current_seq = torch.cat((current_seq[:, 1:, :], pred_tensor), dim=1)
            
    return np.maximum(np.array(predictions), 0.0)
