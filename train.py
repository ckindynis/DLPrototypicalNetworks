from argparse import ArgumentParser
from model.protonet import ProtoNetEncoder
import torch
from constants import Datasets, DistanceMetric

# from dataloader import Dataloader


def train(
    model: ProtoNetEncoder,
    train_dataloader: Dataloader,
    validation_dataloader: Dataloader,
    optimiser: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: str,
    num_epochs: int,
):
    pass


if __name__ == "__main__":
    parser = ArgumentParser

    # Data Arguments
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset to run the experiment.",
        default=Datasets.OMNIGLOT,
        choices=[Datasets.OMNIGLOT, Datasets.MINIIMAGE],
    )

    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the folder that contains the datasets. This should have subfolders for 'omniglot' and 'miniimage' with data already downloaded.",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        help="Path where the experiment assets are saved",
    )

    # Training hyperparams
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs for training",
        default=10,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate for training",
        default=0.001,
    )

    parser.add_argument(
        "--lr_decay_step",
        type=int,
        help="The number of steps after which the learning rate decays",
        default=2000,
    )

    parser.add_argument(
        "--lr_decay_gamma",
        type=float,
        help="Decay factor for the learning rate",
        default=0.5,
    )

    parser.add_argument(
        "--num_episodes",
        type=int,
        help="Number of episodes per epoch",
        default=100,
    )

    parser.add_argument(
        "--num_classes_train",
        type=int,
        help="Number of classes to use in an episode while training. This corresponts to the k in k-way n-shot learning.",
        default=60,
    )

    parser.add_argument(
        "--num_support_train",
        type=int,
        help="Number of support points to use in an episode while training. This corresponts to the n in k-way n-shot learning.",
        default=5,
    )

    parser.add_argument(
        "--num_query_train",
        type=int,
        help="Number of query points to use in an episode while training.",
        default=5,
    )

    parser.add_argument(
        "--num_classes_val",
        type=int,
        help="Number of classes to use in an episode during validation. This corresponts to the k in k-way n-shot learning.",
        default=60,
    )

    parser.add_argument(
        "--num_support_val",
        type=int,
        help="Number of support points to use in an episode during validation. This corresponts to the n in k-way n-shot learning.",
        default=5,
    )

    parser.add_argument(
        "--num_query_val",
        type=int,
        help="Number of query points to use in an episode during validation.",
        default=5,
    )

    # Model Architecture
    parser.add_argument(
        "--conv_kernel_size",
        type=int,
        help="kernel_size x kernel_size filter will be applied to each convolutional layer in the ProtoNet Encoder",
        default=3,
    )

    parser.add_argument(
        "--max_pool_kernel",
        type=int,
        help="max_pool_kernel x max_pool_kernel will be applied to each MaxPool later in each block in the ProtoNet Encoder",
        default=2,
    )

    parser.add_argument(
        "--num_filters",
        type=int,
        help="The number of output filters for each convolutional layer in the ProtoNet Encoder",
        default=64,
    )

    parser.add_argument(
        "--num_conv_layers",
        type=int,
        help="The number of convolutional layer to stack in the ProtoNet Encoder",
        default=4,
    )

    parser.add_argument(
        "--embedding_size",
        type=int,
        help="An optional embedding size for the ProtoNet Encoder. If this is not passed, the output of the final convolutional layer will be flattened to create the embedding",
        required=False,
    )

    parser.add_argument(
        "--distance_metric",
        type=str,
        help="The distance metric to use.",
        default=DistanceMetric.EUCLID,
        choices=[DistanceMetric.EUCLID, DistanceMetric.COSINE],
    )

    args = parser.parse_args()

    cuda_available = torch.cuda.is_available()
    device = "cuda:0" if cuda_available else "cpu"

    if args.dataset == Datasets.OMNIGLOT:
        in_channels = 1
        original_embedding_size = 64
    elif args.dataset == Datasets.MINIIMAGE:
        in_channels = 3
        original_embedding_size = 1600

    model = ProtoNetEncoder(
        in_channels=in_channels,
        out_channels=args.num_filters,
        conv_kernel_size=args.conv_kernel_size,
        num_conv_layers=args.num_conv_layers,
        max_pool_kernel=args.max_pool_kernel,
        original_embedding_size=original_embedding_size,
        new_embedding_size=args.embedding_size,
    ).to(device)

    optimiser = torch.optim.Adam(params=model.params(), lr=args.learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimiser, gamma=args.lr_decay_gamma, step_size=args.lr_decay_step
    )

    # train_dataloader =
    # val_dataloader =
