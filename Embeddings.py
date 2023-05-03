import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim

#%% A function to plot the embeddings in 3D (showing relationships)
def plot_embeddings(words):
    """Plot in 3D the embeddings along with their corresponding words."""
    # Plot the 3D embeddings
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, word in enumerate(words):
        x, y, z = word_embeddings[i].detach().numpy()
        ax.scatter(x, y, z, label=word)
        ax.text(x, y, z, word)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

def get_embedding_from_word(word):
    """Get the embedding vector of a word from the embedding layer"""
    index = [i for i, iword in enumerate(words) if words[i].find(word) != -1][0]
    return word_embeddings[index]

#%% Define the vocabulary size and the embedding dimension
vocab_size = 5
embedding_dim = 3

# Create the embedding layer
torch.manual_seed(0)
embedding_layer = nn.Embedding(vocab_size, embedding_dim)
# Example indices of words in the vocabulary
word_indices = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
# Get the embeddings for the word indices
word_embeddings = embedding_layer(word_indices)

words = ['study', 'party', 'pass', 'fail', 'student']
plot_embeddings(words)

#%%
# Initialize training optimizer and random target embeddings
print(word_embeddings)
# target_embeddings = torch.randn_like(word_embeddings)
target_embeddings = torch.tensor([[0.1, 0.2, 0.3],
                                  [0.1, 0.2, 0.4],
                                  [0.3, 0.4, 0.5],
                                  [0.3, 0.4, 0.6],
                                  [-0.5, 0.6, 0.7]])
# Define the loss function (Mean Squared Error in this case)
loss_function = nn.MSELoss()
optimizer = optim.SGD(embedding_layer.parameters(), lr=0.5)

#%%
# Train the embedding layer
epochs = 1000
for i in range(100):
    optimizer.zero_grad()
    word_embeddings = embedding_layer(word_indices)
    loss = loss_function(word_embeddings, target_embeddings)
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f"Epoch: {i}, Loss: {loss.item()}")
        print(word_embeddings)
        plot_embeddings(words)