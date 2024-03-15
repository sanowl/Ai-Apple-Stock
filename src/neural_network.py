import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation=nn.ReLU(), dropout=0.5):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation = activation
        self.dropout = dropout
        
        # Define input layer
        self.input_layer = nn.Linear(input_size, hidden_layers[0])
        
        # Define hidden layers
        self.hidden = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            self.hidden.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        
        # Define output layer
        self.output_layer = nn.Linear(hidden_layers[-1], output_size)
        
        # Define activation function
        self.activation_function = activation
        
        # Define dropout
        self.dropout_layer = nn.Dropout(p=dropout)
        
        # Define softmax for multi-class classification
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Input layer
        out = self.input_layer(x)
        out = self.activation_function(out)
        out = self.dropout_layer(out)
        
        # Hidden layers
        for layer in self.hidden:
            out = layer(out)
            out = self.activation_function(out)
            out = self.dropout_layer(out)
        
        # Output layer
        out = self.output_layer(out)
        
        # Softmax for multi-class classification
        if self.output_size > 1:
            out = self.softmax(out)
        
        return out
