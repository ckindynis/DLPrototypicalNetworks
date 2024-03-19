import torch

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


def compute_barycenters(support_tensor, num_classes, num_points, emb_size):

    # support_tensor = support points embeddings
    # view as embeddings for each point belonging to each class
    # take mean with respect to the points
    barycenters = support_tensor.view(num_classes, num_points, emb_size).mean(1)

    return barycenters


def proto_loss(X_train, y_train, X_test, y_test, n_support):

    pass