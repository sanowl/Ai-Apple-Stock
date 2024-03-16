import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Load the CSV file
data = pd.read_csv('data/raw/appledata.csv')

# Split features and target variable
X = data.drop(columns=['target'])
y = data['target']

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns

# One-hot encode categorical columns
encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]).toarray(),
                         columns=encoder.get_feature_names_out(categorical_cols))

# Combine encoded columns with numerical columns
X = pd.concat([X_encoded, X.select_dtypes(exclude=['object'])], axis=1)

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        # Input layer
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(nn.Dropout(dropout_rate))

        # Output layer
        self.fc_out = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        for layer in self.hidden_layers:
            out = layer(out)

        out = self.fc_out(out)
        return out

# Define hyperparameters
input_size = X.shape[1]
hidden_sizes = [128, 64]
output_size = len(label_encoder.classes_)
learning_rate = 0.001
num_epochs = 20
batch_size = 64
dropout_rate = 0.3

# Create DataLoader for training and testing data
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize the neural network
model = NeuralNetwork(input_size, hidden_sizes, output_size, dropout_rate)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# Train the model
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}')

    # Compute validation loss
    model.eval()
    val_running_loss = 0.0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_running_loss += loss.item() * inputs.size(0)
    val_loss = val_running_loss / len(test_loader.dataset)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

    # Adjust learning rate based on validation loss
    scheduler.step(val_loss)

# Evaluate the model
model.eval()
predictions = []
true_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.tolist())
        true_labels.extend(labels.tolist())

# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)
print('Accuracy:', accuracy)