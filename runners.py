from datetime import datetime
from time import time

from torch.utils.data import DataLoader, Dataset

from model.protonet import ProtoNetEncoder
import torch
from constants import TensorboardAssets, AssetNames
from dataloader import DatasetBase
from tqdm import tqdm
from model.helpers import protoLoss, EarlyStopper, dataloader_batch_removal_collate_fn
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter


def validation(
    model: ProtoNetEncoder,
    validation_dataset: Dataset,
    device: str,
    num_support_val: int,
    num_query_val: int,
    distance_metric: str,
):
    val_losses = []
    val_accuracies = []

    model.eval()  # Needed, as we are using BatchNorm2d (see https://pytorch.org/docs/stable/notes/autograd.html#evaluation-mode-nn-module-eval)
    with torch.inference_mode():  # See: https://pytorch.org/docs/stable/notes/autograd.html#inference-mode. If errors occur, change this to torch.no_grad()
        val_episode_loader = DataLoader(validation_dataset, batch_size=1, collate_fn=dataloader_batch_removal_collate_fn)
        for episode_num, (val_image_tensors, val_labels) in enumerate(val_episode_loader):
            val_image_tensors, val_labels = val_image_tensors.to(device), val_labels.to(device)
            val_embeddings = model(val_image_tensors)

            val_loss, val_acc = protoLoss(
                model_output=val_embeddings,
                target_output=val_labels,
                n_support=num_support_val,
                n_query=num_query_val,
                distance_metric=distance_metric,
            )

            val_losses.append(val_loss.item())
            val_accuracies.append(val_acc.item())

    return val_losses, val_accuracies


def test(
    model: ProtoNetEncoder,
    test_dataset: DatasetBase,
    device: str,
    num_support_test: int,
    num_query_test: int,
    distance_metric: str,
):
    print("Running testing...")
    test_accuracy = []

    model.eval()
    with torch.inference_mode():
        for test_batch in test_dataset:
            test_image_tensors, test_labels = test_batch
            test_image_tensors, test_labels = test_image_tensors.to(
                device
            ), test_labels.to(device)
            test_embeddings = model(test_image_tensors)

            _, test_acc = protoLoss(
                model_output=test_embeddings,
                target_output=test_labels,
                n_support=num_support_test,
                n_query=num_query_test,
                distance_metric=distance_metric,
            )
            test_accuracy.append(test_acc.item())

    avg_accuracy = np.mean(test_accuracy)

    print(f"Final test Accuracy is {avg_accuracy}")


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
    distance_metric: str,
    save_path: str,
    writer: SummaryWriter,
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
        train_episode_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=dataloader_batch_removal_collate_fn)
        for episode_num, (image_tensors, label_tensors) in enumerate(
                tqdm(
                    train_episode_loader,
                    desc=f"Doing episodes for epoch {epoch + 1}",
                    total=num_episodes_per_epoch,
                )
        ):
            model.train()

            optimiser.zero_grad()
            num_steps += 1
            start_time = time()
            image_tensors, label_tensors = image_tensors.to(device), label_tensors.to(device)
            print(f"Time taken to move tensors to device: {time() - start_time}")

            start_time = time()
            embeddings = model(image_tensors)
            print(f"Time taken to get embeddings: {time() - start_time}")

            start_time = time()
            train_loss, train_acc = protoLoss(
                model_output=embeddings,
                target_output=label_tensors,
                n_support=num_support_train,
                n_query=num_query_train,
                distance_metric=distance_metric,
            )
            print(f"Time taken to calculate loss: {time() - start_time}")
            start_time = time()
            train_loss.backward()
            print(f"Time taken to backpropagate: {time() - start_time}")

            start_time = time()
            optimiser.step()
            print(f"Time taken to step optimiser: {time() - start_time}")
            print("-"* 20)
            train_losses.append(train_loss.item())
            train_accuracies.append(train_acc.item())

            if num_steps % num_validation_steps == 0:
                print(
                    f"\nPerforming validation at epoch {epoch + 1} and episode {episode_num + 1}"
                )

                val_losses_run, val_accuracies_run = validation(
                    model=model,
                    validation_dataset=validation_dataset,
                    device=device,
                    num_support_val=num_support_val,
                    num_query_val=num_query_val,
                    distance_metric=distance_metric,
                )

                val_losses.extend(val_losses_run)
                val_accuracies.extend(val_accuracies_run)

                avg_val_acc = round(
                    np.mean(val_accuracies[-num_episodes_per_epoch:]), 3
                )
                avg_val_loss = np.mean(val_losses[-num_episodes_per_epoch:])
                
                writer.add_scalar(TensorboardAssets.VAL_ACC, avg_val_acc, num_steps)
                writer.add_scalar(TensorboardAssets.VAL_LOSS, avg_val_loss, num_steps)

                print(
                    f"Epoch {epoch + 1} | Validation Accuracy: {avg_val_acc} | Validation Loss: {avg_val_loss}"
                )

                # TODO activate this later
                # if early_stopping.early_stop(avg_val_loss):
                #     print(
                #         f"Early stopping at epoch {epoch + 1} with validation loss {avg_val_loss}"
                #     )
                #     break

                if best_val_loss > avg_val_loss:
                    print(
                        f"New validation loss {avg_val_loss} is better than previous val loss {best_val_loss}. Saving model."
                    )
                    best_val_loss = avg_val_loss
                    best_state = model.state_dict()
                    torch.save(model.state_dict(), save_model_path.replace(
                        ".pth", f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.pth"
                    ),)

            lr_scheduler.step()

        avg_train_acc = round(np.mean(train_accuracies[-num_episodes_per_epoch:]), 3)
        avg_train_loss = np.mean(train_losses[-num_episodes_per_epoch:])
        
        writer.add_scalar(TensorboardAssets.TRAIN_ACC, avg_train_acc, epoch+1)
        writer.add_scalar(TensorboardAssets.TRAIN_LOSS, avg_train_loss, epoch+1)

        print(
            f"Epoch {epoch + 1} | Train Accuracy: {avg_train_acc} | Train Loss: {avg_train_loss}"
        )

    print("Training completed!")
    return best_state, train_accuracies, train_losses, val_accuracies, val_losses
