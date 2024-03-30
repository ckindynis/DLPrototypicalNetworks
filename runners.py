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
