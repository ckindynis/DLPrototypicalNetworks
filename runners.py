from argparse import ArgumentParser
from datetime import datetime

from model.protonet import ProtoNetEncoder
import torch
from constants import Datasets, DistanceMetric, Modes, AssetNames
from dataloader import DatasetBase, MiniImageNetDataset
from tqdm import tqdm
from model.helpers import protoLoss, EarlyStopper
import numpy as np
import os


def train(
    model: ProtoNetEncoder,
    train_dataset: DatasetBase,
    validation_dataset: DatasetBase,
    optimiser: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: str,
    num_epochs: int,
    num_episodes_per_epoch: int,
    num_support_train: int,
    num_query_train: int,
    num_support_val: int,
    num_query_val: int,
    num_validation_steps: int,
    early_stopping_patience: int,
    early_stopping_delta: float,
    save_path: str,
):

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float("inf")
    best_state = None

    early_stopping = EarlyStopper(
        patience=early_stopping_patience, min_delta=early_stopping_delta
    )
    save_model_path = os.path.join(save_path, AssetNames.MODEL)

    num_steps = 0
    for epoch in range(num_epochs):
        for episode_num, episode in enumerate(tqdm(train_dataset)):
            print(f"Training epoch {epoch + 1} and episode {episode_num + 1}")
            model.train()
            num_steps += 1
            image_tensors, label_tensors = episode
            image_tensors, label_tensors = image_tensors.to(device), label_tensors.to(
                device
            )

            embeddings = model(image_tensors)

            train_loss, train_acc = protoLoss(
                model_output=embeddings,
                target_output=label_tensors,
                n_support=num_support_train,
                n_query=num_query_train,
            )
            train_loss.backward()

            optimiser.step()

            train_losses.append(train_loss.item())
            train_accuracies.append(train_acc.item())

            if num_steps % num_validation_steps == 0:
                print(
                    f"Performing validation at epoch {epoch + 1} and episode {episode_num + 1}"
                )

                model.eval()  # Needed, as we are using BatchNorm2d (see https://pytorch.org/docs/stable/notes/autograd.html#evaluation-mode-nn-module-eval)

                with torch.inference_mode():    # See: https://pytorch.org/docs/stable/notes/autograd.html#inference-mode. If errors occur, change this to torch.no_grad()
                    for val_batch in validation_dataset:
                        val_image_tensors, val_labels = val_batch
                        val_image_tensors, val_labels = val_image_tensors.to(
                            device
                        ), val_labels.to(device)
                        val_embeddings = model(val_image_tensors)

                        val_loss, val_acc = protoLoss(
                            model_output=val_embeddings,
                            target_output=val_labels,
                            n_support=num_support_val,
                            n_query=num_query_val,
                            distance_metric=DistanceMetric.EUCLID,
                        )

                        val_losses.append(val_loss.item())
                        val_accuracies.append(val_acc.item())

                    avg_val_acc = round(
                        np.mean(val_accuracies[-num_episodes_per_epoch:]), 3
                    )
                    avg_val_loss = np.mean(val_losses[-num_episodes_per_epoch:])

                    print(
                        f"Epoch {epoch + 1} | Validation Accuracy: {avg_val_acc} | Validation Loss: {avg_val_loss}"
                    )

                    # TODO activate this later
                    # if early_stopping.early_stop(avg_val_loss):
                    #     print(
                    #         f"Early stopping at epoch {epoch + 1} with validation loss {avg_val_loss}"
                    #     )
                    #     break

                    if best_val_loss > avg_val_acc:
                        print(
                            f"New validation loss {avg_val_loss} is better than previous val loss {best_val_loss}. Saving model."
                        )
                        best_val_loss = avg_val_loss
                        best_state = model.state_dict()
                        # torch.save(model.state_dict(), save_model_path)
                    torch.save(model.state_dict(), save_model_path.replace(".pth", f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.pth"))  # TODO: for now always saving the models

        avg_train_acc = round(np.mean(train_accuracies[-num_episodes_per_epoch:]), 3)
        avg_train_loss = np.mean(train_accuracies[-num_episodes_per_epoch:])

        print(f"Epoch {epoch + 1} | Train Accuracy: {avg_train_acc} | Train loss Loss: {avg_train_loss}")
        lr_scheduler.step()
    
    print("Training completed!")
    return best_state, train_accuracies, train_losses, val_accuracies, val_losses


if __name__ == "__main__":
    parser = ArgumentParser()

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
        "--num_validation_steps",
        type=int,
        help="Number of steps after which you conduct validation",
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
        default=5,
    )

    parser.add_argument(
        "--num_support_train",
        type=int,
        help="Number of support points to use in an episode while training. This corresponts to the n in k-way n-shot learning.",
        default=20,
    )

    parser.add_argument(
        "--num_query_train",
        type=int,
        help="Number of query points to use in an episode while training.",
        default=10,
    )

    parser.add_argument(
        "--num_classes_val",
        type=int,
        help="Number of classes to use in an episode during validation. This corresponts to the k in k-way n-shot learning.",
        default=5,
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

    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        help="Patience for early stopping",
        default=3,
    )

    parser.add_argument(
        "--early_stopping_delta",
        type=float,
        help="Delta for early stopping",
        default=0.05,
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

    optimiser = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimiser, gamma=args.lr_decay_gamma, step_size=args.lr_decay_step
    )

    if args.dataset == Datasets.MINIIMAGE:
        train_dataset = MiniImageNetDataset(
            base_dir=args.data_path,
            k_way=args.num_classes_train,
            k_shot=args.num_support_train,
            k_query=args.num_query_train,
            n_episodes=args.num_episodes,
        )
        val_dataset = MiniImageNetDataset(
            base_dir=args.data_path,
            k_way=args.num_classes_val,
            k_shot=args.num_support_val,
            k_query=args.num_query_val,
            n_episodes=args.num_episodes,
            mode=Modes.VAL,
        )

    # Add For Omniglot here

    best_state, train_accuracies, train_losses, val_accuracies, val_losses = train(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        optimiser=optimiser,
        lr_scheduler=lr_scheduler,
        device=device,
        num_epochs=args.num_epochs,
        num_episodes_per_epoch=args.num_episodes,
        num_query_train=args.num_query_train,
        num_support_train=args.num_support_train,
        num_query_val=args.num_query_val,
        num_support_val=args.num_support_val,
        num_validation_steps=args.num_validation_steps,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_delta=args.early_stopping_delta,
        save_path=args.save_path,
    )