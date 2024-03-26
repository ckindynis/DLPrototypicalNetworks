from torch import nn
from typing import Optional
import torch


class ConvBlock(nn.module):
    def __init__(
        self,
        in_channels,
        out_channels: int,
        conv_kernel_size: int,
        max_pool_kernel: int,
    ):
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=conv_kernel_size,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=max_pool_kernel),
        )

    def forward(self, x):
        out = self.conv_block(x)
        return out


class ProtoNetEncoder(nn.module):
    def __init__(
        self,
        in_channels,
        out_channels: int,
        conv_kernel_size,
        max_pool_kernel: int,
        num_conv_layers: int,
        original_embedding_size: Optional[int] = None,
        new_embedding_size: Optional[int] = None,
    ):
        layers = []

        layers.append(
            ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=conv_kernel_size,
                max_pool_kernel=max_pool_kernel,
            )
        )

        for _ in range(num_conv_layers - 1):
            layers.append(
                ConvBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=conv_kernel_size,
                    max_pool_kernel=max_pool_kernel,
                )
            )

        self.encoding_block = nn.Sequential(*layers)

        self.embedding_layer = None
        if new_embedding_size:
            self.embedding_layer = nn.Linear(
                original_embedding_size, new_embedding_size
            )

    def forward(self, x):
        embedding = self.encoding_block(x)

        embedding = embedding.view(embedding.size(0), -1)
        if self.embedding_layer:
            embedding = self.embedding_layer(embedding)

        return embedding
        

def euclidean_distance(tensor1, tensor2):
    # Expand dimensions to allow broadcasting
    expanded_tensor1 = tensor1.unsqueeze(1)
    expanded_tensor2 = tensor2.unsqueeze(0)
    
    # Calculate the squared Euclidean distance
    squared_diff = (expanded_tensor1 - expanded_tensor2)**2
    squared_distances = squared_diff.sum(dim=-1)
    
    # Take the square root to get the Euclidean distance
    distances = torch.sqrt(squared_distances)
    
    return distances

def protoLoss(model_output, target_output, n_support, n_query):

    """
    Args:   model_output: torch.Tensor of shape (n_support+n_query, d), where d is the embedding dimension
            target_output: torch.Tensor of shape (n_support+n_query), containing the target indices
            n_support: number of examples per class in the support set
            n_query: number of examples per class in the query set
    """

    # Get unique classes in the episode
    episode_classes = torch.unique(target_output)

    # Extract support and query points for each class
    support_points = []
    query_points = []
    query_targets = []
    for c in episode_classes:
        support_points.append(model_output[target_output == c][:n_support])
        query_points.append(model_output[target_output == c][n_support:n_support+n_query])
        query_targets.append(target_output[target_output == c][n_support:n_support+n_query])

    # View each query point independently
    query_points = torch.stack(query_points).view(-1, len(episode_classes)*n_query)
    query_targets = torch.stack(query_targets).view(-1, len(episode_classes)*n_query)

    # Calculate barycenters for each class
    barycenters = torch.stack([torch.mean(s, dim=0) for s in support_points])

    # Init loss
    loss = 0
    acc = 0

    # Calculate loss for each query point
    for  i, q in enumerate(query_points):
        distance_list = []
        exp_sum = 0
        # Calculate distance to each barycenter
        for c in range(len(episode_classes)):

            distances = torch.sum(euclidean_distance(q, barycenters[c]))
            distance_list.append(torch.exp(-distances))
            exp_sum += torch.exp(-distances)

        # Calculate log softmax of distances
        log_smax = torch.log_softmax(torch.stack(distance_list)/exp_sum, dim=0)

        # Calculate accuracy
        pred_output = torch.argmax(log_smax)
        acc += (pred_output.item() == query_targets[0][i])

        # Calculate loss for the query point, for true class
        loss -= log_smax[query_targets[0][i]]

    # Return loss and normalized accuracy
    return loss, acc/(n_query*len(episode_classes))



def generate_sample_data(n_classes=5, n_support=3, n_query=2, embedding_dim=10):
    # Calculate total number of examples
    n_total = n_support + n_query

    # Initialize tensors for model_output and target_output
    model_output = torch.empty(n_classes * n_total, embedding_dim)
    target_output = torch.empty(n_classes * n_total, dtype=torch.long)

    # Generate sample data for each class
    for i in range(n_classes):
        # Generate random embeddings for support and query examples
        class_embeddings = torch.randn(n_total, embedding_dim)
        
        # Assign embeddings to model_output tensor
        model_output[i * n_total: (i + 1) * n_total] = class_embeddings
        
        # Assign target indices to target_output tensor
        target_output[i * n_total: i * n_total + n_support] = i  # Assigning support set indices
        target_output[(i + 1) * n_total - n_query: (i + 1) * n_total] = i  # Assigning query set indices

    print("Model Output Shape:", model_output.shape)
    print("Target Output Shape:", target_output.shape)
    return model_output, target_output
