import torch
import torch.nn as nn

# Define the size of the input tensor and the embedding dimension
num_embeddings = 1
embedding_dim = 32

# Generate some sample data as a tensor of indices
input_tensor = torch.randint(0, num_embeddings, (1, 1000))
print(input_tensor.shape)

# Define the embedding layer
embedding_layer = nn.Embedding(num_embeddings, embedding_dim)

# Apply the embedding layer to the input tensor
embedded_tensor = embedding_layer(input_tensor)

# Take the mean along the second dimension to get a 1-by-32 tensor

# Print the shape of the resulting tensor
print(embedded_tensor.shape)
