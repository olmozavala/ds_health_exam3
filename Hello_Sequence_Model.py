#%%
import torch
import torch.nn as nn
torch.manual_seed(0)

# This is a simple model that learns to predict the 'next' character in a sequence of characters.
# The model is a simple RNN with one hidden layer.
# The input is a sequence of characters, and the output is a probability distribution over the possible characters.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu" # IN this case is faster in the CPU
print(f"Using device: {device}")

# Define the RNN model
class CharRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, output_size, n_layers=1):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(input_size, emb_size)
        self.rnn = nn.GRU(emb_size, hidden_size, n_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embed(input)
        output, hidden = self.rnn(embedded.view(1, 1, -1), hidden)
        output = self.fc(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size)

#%%
# Dataset preparation
text = "Final exam, you got this! RNNs are powerful models that can be used for many tasks."
chars = list(set(text))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

input_size = len(chars)
hidden_size = 20
emb_size = 8
output_size = len(chars)
n_layers = 1

model = CharRNN(input_size, emb_size, hidden_size, output_size, n_layers).to(device)

# Training the model
learning_rate = 0.005
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_epochs = 500

#%%
for epoch in range(1, n_epochs + 1):
    input_seq = torch.tensor([char_to_idx[ch] for ch in text[:-1]], dtype=torch.long).to(device)
    target_seq = torch.tensor([char_to_idx[ch] for ch in text[1:]], dtype=torch.long).to(device)

    hidden = model.init_hidden().to(device)
    model.zero_grad()
    loss = 0

    for i in range(input_seq.size(0)):
        output, hidden = model(input_seq[i], hidden)
        loss += criterion(output, target_seq[i].view(1))

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item() / input_seq.size(0)}")

#%%
# Test the model
start_str = "Final exam, you got this! RNNs are po"

hidden = model.init_hidden()
input_seq = torch.tensor([char_to_idx[ch] for ch in start_str], dtype=torch.long)

full_pred = ""
print(f"Input: {start_str}")
print("Output:" , end=" ")
for i in range(len(start_str)):
    output, hidden = model(input_seq[i], hidden)
    full_pred += idx_to_char[output.argmax().item()]
    print(idx_to_char[output.argmax().item()], end="")