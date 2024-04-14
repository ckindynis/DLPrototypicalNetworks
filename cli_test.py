import torch
import typer

from constants import Datasets, DistanceMetric, Modes, AssetNames
from dataloader import MiniImageNetDataset
from runners import test
import os
from configs import configuration, DatasetConfiguration
from model.protonet import ProtoNetEncoder

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def run_experiment(
    dataset: Datasets = typer.Option(
        Datasets.MINIIMAGE, help="Name of the dataset to run the experiment."
    ),
    data_path: str = typer.Option(
        ..., help="Path to the folder that contains the datasets."
    ),
    save_path: str = typer.Option(
        ..., help="Path where the experiment assets are saved."
    ),
    num_classes_test: int = typer.Option(
        5, help="Number of classes to use in an episode during testing."
    ),
    num_support_test: int = typer.Option(
        5, help="Number of support points to use in an episode during testing."
    ),
    num_query_test: int = typer.Option(
        15, help="Number of query points to use in an episode during testing."
    ),
    num_episodes_test: int = typer.Option(10, help="Number of episodes to test on."),
    distance_metric: DistanceMetric = typer.Option(
        DistanceMetric.EUCLID, help="The distance metric to use."
    ),
    conv_kernel_size: int = typer.Option(
        3, help="Kernel size for the convolutional layers in the ProtoNet Encoder."
    ),
    max_pool_kernel: int = typer.Option(
        2, help="Kernel size for max pooling in the ProtoNet Encoder."
    ),
    num_filters: int = typer.Option(
        64, help="The number of output filters for each convolutional layer."
    ),
    num_conv_layers: int = typer.Option(
        4, help="The number of convolutional layers in the ProtoNet Encoder."
    ),
    embedding_size: int = typer.Option(
        None,
        help="An optional embedding size for the ProtoNet Encoder.",
        show_default=False,
    ),
):
    device = "cpu"

    if dataset == Datasets.MINIIMAGE:
        test_dataset = MiniImageNetDataset(
            base_dir=data_path,
            k_way=num_classes_test,
            k_shot=num_support_test,
            k_query=num_query_test,
            n_episodes=num_episodes_test,
            mode=Modes.TEST,
        )
        in_channels = 3
        original_embedding_size = 1600

    model = ProtoNetEncoder(
        in_channels=in_channels,
        out_channels=num_filters,
        conv_kernel_size=conv_kernel_size,
        max_pool_kernel=max_pool_kernel,
        num_conv_layers=num_conv_layers,
        original_embedding_size=original_embedding_size,
        new_embedding_size=embedding_size,
    )
    model_path = os.path.join(save_path, AssetNames.MODEL)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    test(
        model=model,
        test_dataset=test_dataset,
        device=device,
        num_query_test=num_query_test,
        num_support_test=num_support_test,
        distance_metric=distance_metric,
    )


if __name__ == "__main__":
    app()
