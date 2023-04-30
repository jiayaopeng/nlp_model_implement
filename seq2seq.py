import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # output shape: (batch_size, seq_len, hidden_size)
        # hn shape: (num_layers, batch_size, hidden_size)
        # cn shape: (num_layers, batch_size, hidden_size)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        return output, hn, cn

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hn, cn):
        # x shape: (batch_size, 1, input_size)
        # hn shape: (num_layers, batch_size, hidden_size)
        # cn shape: (num_layers, batch_size, hidden_size)
        output, (hn, cn) = self.lstm(x, (hn, cn))

        # output shape: (batch_size, 1, hidden_size)
        # pred shape: (batch_size, 1, output_size)
        pred = self.fc(output)
        return pred, hn, cn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y):
        # x shape: (batch_size, seq_len, input_size)
        # y shape: (batch_size, seq_len, output_size)
        batch_size, seq_len, output_size = y.size()

        # Initialize outputs with zeros
        outputs = torch.zeros(batch_size, seq_len, output_size).to(y.device)

        # Get encoder output and hidden states
        encoder_output, hn, cn = self.encoder(x)

        # Set the decoder input to be the start token
        decoder_input = torch.zeros(batch_size, 1, output_size).to(y.device)

        for i in range(seq_len):
            # Get decoder output and hidden states
            decoder_output, hn, cn = self.decoder(decoder_input, hn, cn)

            # Store decoder output
            outputs[:, i, :] = decoder_output.squeeze(1)

            # Set the decoder input to be the current prediction
            decoder_input = y[:, i, :].unsqueeze(1)

        return outputs

# Define hyperparameters
input_size = 100
output_size = 200
hidden_size = 128
num_layers = 2
learning_rate = 0.001
batch_size = 32
num_epochs = 10

# Initialize the model
encoder = Encoder(input_size, hidden_size, num_layers)
decoder = Decoder(output_size, hidden_size, output_size, num_layers)
model = Seq2Seq(encoder, decoder)


# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define dummy dataset
class DummyDataset(Dataset):
    def init(self, size):
        self.data = torch.randn(size, 10, input_size)
        self.target = torch.randn(size, 10, output_size)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return len(self.data)
    

#Initialize data loader
dataset = DummyDataset(100)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#Train the model
for epoch in range(num_epochs):
    for i, (x, y) in enumerate(dataloader):
    # Zero the gradients
        optimizer.zero_grad()

    # Forward pass
    outputs = model(x, y)

    # Compute loss and gradients
    loss = criterion(outputs, y)
    loss.backward()

    # Update parameters
    optimizer.step()

    if (i+1) % 10 == 0:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(dataloader), loss.item()))
        
#Test the model
test_x = torch.randn(1, 10, input_size)
predicted_y = model(test_x, test_x)
print('Input:', test_x)
print('Output:', predicted_y)







