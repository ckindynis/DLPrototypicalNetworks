import torch
from constants import DistanceMetric


def euclidean_distance(tensor1, tensor2):
    # Expand dimensions to allow broadcasting
    expanded_tensor1 = tensor1.unsqueeze(1)
    expanded_tensor2 = tensor2.unsqueeze(0)

    # Calculate the squared Euclidean distance
    squared_diff = (expanded_tensor1 - expanded_tensor2) ** 2
    squared_distances = squared_diff.sum(dim=-1)

    # Take the square root to get the Euclidean distance
    distances = torch.sqrt(squared_distances)

    return distances


def protoLoss(
    model_output: torch.Tensor,
    target_output: torch.Tensor,
    n_support: int,
    n_query: int,
    distance_metric: str = DistanceMetric.EUCLID,
):
    """
    Args:   model_output: torch.Tensor of shape (n_support+n_query, d), where d is the embedding dimension
            target_output: torch.Tensor of shape (n_support+n_query), containing the target indices
            n_support: number of examples per class in the support set
            n_query: number of examples per class in the query set
            distance_metric: The distance metric used to calculate distance between query and the barycenters
    """

    # Get unique classes in the episode
    episode_classes = torch.unique(target_output)

    # Extract support and query points for each class
    support_points = []
    query_points = []
    query_targets = []
    for c in episode_classes:
        support_points.append(model_output[target_output == c][:n_support])
        query_points.append(
            model_output[target_output == c][n_support : n_support + n_query]
        )
        query_targets.append(
            target_output[target_output == c][n_support : n_support + n_query]
        )

    # View each query point independently
    query_points = torch.stack(query_points).view(-1, len(episode_classes) * n_query)
    query_targets = torch.stack(query_targets).view(-1, len(episode_classes) * n_query)

    # Calculate barycenters for each class
    barycenters = torch.stack([torch.mean(s, dim=0) for s in support_points])

    # Init loss
    loss = 0
    acc = 0

    # Calculate loss for each query point
    for q_idx in range(query_points.shape[1]):
        distance_list = []
        exp_sum = 0
        # Calculate distance to each barycenter
        for c in range(len(episode_classes)):

            if distance_metric == DistanceMetric.EUCLID:
                distances = torch.cdist(
                    query_points[:, q_idx].unsqueeze(0), barycenters[c].unsqueeze(0)
                )
            # Add for other distance metrics
            distance_list.append(torch.exp(-distances))
            exp_sum += torch.exp(-distances)

        # import pdb; pdb.set_trace()
        # Calculate log softmax of distances
        log_smax = torch.log_softmax(torch.stack(distance_list) / exp_sum, dim=0)

        # Calculate accuracy
        pred_output = torch.argmax(log_smax)
        acc += pred_output.item() == query_targets[0][q_idx]

        # TODO: Fix this mess later
        c_idx = torch.where(episode_classes == query_targets[0][q_idx].item())[0].item()

        # Calculate loss for the query point, for true class

        loss -= log_smax[c_idx]

    # Return loss and normalized accuracy
    return loss, acc / (n_query * len(episode_classes))


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False