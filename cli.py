import click
import torch
import typer

from constants import Datasets, DistanceMetric, Modes
from dataloader import MiniImageNetDataset
from model.protonet import ProtoNetEncoder
from runners import train, test
from configs import configuration, DatasetConfiguration
from torch.utils.tensorboard import SummaryWriter
import os

app = typer.Typer()


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
    num_epochs: int = typer.Option(100, help="Number of epochs for training."),
    num_episodes_train: int = typer.Option(None, help="Number of episodes per epoch for training."),
    num_episodes_test: int = typer.Option(None, help="Number of episodes to test on."),
    num_validation_steps: int = typer.Option(
        100, help="Number of steps after which you conduct validation."
    ),
    learning_rate: float = typer.Option(None, help="Learning rate for training."),
    lr_decay_step: int = typer.Option(
        None, help="The number of steps after which the learning rate decays."
    ),
    lr_decay_gamma: float = typer.Option(
        None, help="Decay factor for the learning rate."
    ),
    num_classes_train: int = typer.Option(
        None, help="Number of classes to use in an episode while training."
    ),
    num_support_train: int = typer.Option(
        None, help="Number of support points to use in an episode while training."
    ),
    num_query_train: int = typer.Option(
        None, help="Number of query points to use in an episode while training."
    ),
    num_classes_val: int = typer.Option(
        None, help="Number of classes to use in an episode during validation."
    ),
    num_support_val: int = typer.Option(
        None, help="Number of support points to use in an episode during validation."
    ),
    num_query_val: int = typer.Option(
        None, help="Number of query points to use in an episode during validation."
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
    distance_metric: DistanceMetric = typer.Option(
        DistanceMetric.EUCLID, help="The distance metric to use."
    ),
    early_stopping_patience: int = typer.Option(3, help="Patience for early stopping."),
    early_stopping_delta: float = typer.Option(0.05, help="Delta for early stopping."),
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Taking default for dataset if these arguments are not passed
    dataset_configuration: DatasetConfiguration = configuration[dataset]
    num_episodes_train = num_episodes_train or dataset_configuration.num_episodes_train
    num_episodes_test = num_episodes_test or dataset_configuration.num_episodes_test
    learning_rate = learning_rate or dataset_configuration.learning_rate
    lr_decay_step = lr_decay_step or dataset_configuration.lr_decay_step
    lr_decay_gamma = lr_decay_gamma or dataset_configuration.lr_decay_gamma
    num_classes_train = num_classes_train or dataset_configuration.num_classes_train
    num_support_train = num_support_train or dataset_configuration.num_support_train
    num_query_train = num_query_train or dataset_configuration.num_query_train
    num_classes_val = num_classes_val or dataset_configuration.num_classes_val
    num_support_val = num_support_val or dataset_configuration.num_support_val
    num_query_val = num_query_val or dataset_configuration.num_query_val
    
    tensorboard_log_dir = os.path.join(save_path, "Tensorboard")
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    
    if dataset == Datasets.OMNIGLOT:
        in_channels = 1
        original_embedding_size = 64
    elif dataset == Datasets.MINIIMAGE:
        in_channels = 3
        original_embedding_size = 1600

        model = ProtoNetEncoder(
            in_channels=in_channels,
            out_channels=num_filters,
            conv_kernel_size=conv_kernel_size,
            num_conv_layers=num_conv_layers,
            max_pool_kernel=max_pool_kernel,
            original_embedding_size=original_embedding_size,
            new_embedding_size=embedding_size,
        ).to(device)

        optimiser = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimiser, gamma=lr_decay_gamma, step_size=lr_decay_step
        )

        if dataset == Datasets.MINIIMAGE:
            train_dataset = MiniImageNetDataset(
                base_dir=data_path,
                k_way=num_classes_train,
                k_shot=num_support_train,
                k_query=num_query_train,
                n_episodes=num_episodes_train,
            )
            val_dataset = MiniImageNetDataset(
                base_dir=data_path,
                k_way=num_classes_val,
                k_shot=num_support_val,
                k_query=num_query_val,
                n_episodes=num_episodes_train,
                mode=Modes.VAL,
            )
            test_dataset = MiniImageNetDataset(
                base_dir=data_path,
                k_way=num_classes_val,
                k_shot=num_support_val,
                k_query=num_query_val,
                n_episodes=num_episodes_test,
                mode=Modes.TEST,
            )

        # TODO Add For Omniglot here

        best_state, train_accuracies, train_losses, val_accuracies, val_losses = train(
            model=model,
            train_dataset=train_dataset,
            validation_dataset=val_dataset,
            optimiser=optimiser,
            lr_scheduler=lr_scheduler,
            device=device,
            num_epochs=num_epochs,
            num_episodes_per_epoch=num_episodes_train,
            num_query_train=num_query_train,
            num_support_train=num_support_train,
            num_query_val=num_query_val,
            num_support_val=num_support_val,
            num_validation_steps=num_validation_steps,
            early_stopping_patience=early_stopping_patience,
            early_stopping_delta=early_stopping_delta,
            save_path=save_path,
            distance_metric=distance_metric,
            writer=writer
        )

        test(
            model=model,
            test_dataset=test_dataset,
            device=device,
            num_query_test=num_query_val,
            num_support_test=num_support_val,
            distance_metric=distance_metric,
        )
        
        writer.close()


if __name__ == "__main__":
    app()
