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
    model_output = model_output.to('cpu')
    target_output = target_output.to('cpu')

    # Get unique classes in the episode
    episode_classes = torch.unique(target_output)

    reshaped_tensor = model_output.view(-1, n_support+n_query, model_output.size(1))

    support_points = reshaped_tensor[:, :n_support, :]
    query_points = reshaped_tensor[:, n_support:, :].reshape(len(episode_classes)*n_query, model_output.size(1))
    query_targets = target_output.view(-1, n_support+n_query)[:, n_support:].reshape(1, len(episode_classes)*n_query)

    barycenters = support_points.mean(dim=1)

    query_indexes = [i // n_query for i in range(len(query_targets[0]))]

    ds = torch.cdist(query_points, barycenters)

    log_smax = torch.log_softmax(-ds, dim=1)
    
    pred_output = torch.argmax(log_smax, dim=1)

    loss = 0
    loss = -log_smax[torch.arange(log_smax.size(0)), torch.tensor(query_indexes)].mean()

    acc = torch.sum(episode_classes[pred_output] == query_targets[0]).item() / len(query_targets[0]*len(episode_classes))

    return loss, torch.tensor(acc)


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


def dataloader_batch_removal_collate_fn(batch):
    # Since we're doing single episodes, batch should have a single element
    data, labels = batch[0]
    # No need to modify data or labels, as we're not really batching
    return data, labels