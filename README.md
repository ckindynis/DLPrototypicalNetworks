# Prototypical Networks
This repository is a reproduction of the [Prototypical Networks for Few-shot Learning](https://arxiv.org/pdf/1703.05175) paper by Snell et al.

# Introduction
Few-shot learning is a highly researched topic in the machine learning community. Making classification decisions for unseen classes only by using a few examples is highly valuable, especially in applications where data collection is difficult or expensive.

The Prototypical Networks proposed by Snell et al. introduce a novel approach to address this problem by learning a metric space where classification is performed based on distances to prototype representations of each class. In this repository, we provide a simpler and more modular implementation of the original implementation and specifically focus on the mini-ImageNet dataset since it is harder to train.

# Background

At its core, Prototypical Networks learn a non-linear mapping of the input data into an embedding space using a neural network. In this embedding space, each class is represented by a prototype, which is calculated as the barycentric mean of the support set examples belonging to that class.

The process involves two types of data points:

1. **Support Points:** These are used to calculate the prototype for each class. The barycentric mean of the embeddings of the support points form the class embedding. The support set consists of a few labeled examples per class that are chosen in an episode.
2. **Query Points:** These are data points that need to be classified into one of the classes chosen for the episode, serving as the input data to learn the embedding. The objective is to ensure that points from the same class are classified nearby in the embedding space.

![ Few-Shot Learning with Prototypical Networks](https://hackmd-prod-images.s3-ap-northeast-1.amazonaws.com/uploads/upload_2f69c48cf0a863b5ff232ab17526128d.png?AWSAccessKeyId=AKIA3XSAAW6AWSKNINWO&Expires=1714909782&Signature=IUsicDEtcV3%2FlE0RG1H2Zp%2BjJqQ%3D)

Prototypical Networks can be related to clustering; classes are represented as clusters in the embedding space, and the prototype serves as the centroid of each cluster. This clustering approach enables efficient classification of query points by measuring their distance to the class prototypes. In their paper, Snell et al. found that Euclidean distance performs better than cosine similarity as a distance metric in the embedding space and the same metric is therefore used in our experiments as well.

Training Prototypical Networks involves forming training episodes by randomly selecting a set of classes. For each class, a subset of examples is chosen as the support set, while a subset of the remaining examples is used as query points. The model is then trained by minimizing the negative log probability of the true class using Stochastic Gradient Descent. Overall, Prototypical Networks offer an effective and efficient approach to few-shot learning, leveraging the concept of class prototypes and embedding spaces to generalize to new classes with limited labeled examples.

# Our Replication
As mentioned before, the focus of this replication is to replicate the results from the original paper doing experiments on the mini-Imagenet dataset. This dataset consists 60,000 color images with 600 examples per class. Following the original paper, the dataset has been split on a class basis to obtain 64 classes for training, 16 classes for validating, and 20 classes for testing. We are using the same number of classes for each set, but we are not using the same class division as the original paper. The goal is to see if the model performance indicated in the paper can also be achieved with another data distribution, this way not only replicating but also reevaluating the work in the original paper. The only preprocessing necessary for these images is resizing them to 84 × 84 images (using bilinear interpolation).

# Experiments and Results
In our experiments, we evaluated the performance of our Prototypical Networks model on the mini-ImageNet dataset. We conducted experiments for various few-shot learning scenarios, including 30-way 1-shot learning and 20-way 5-shot learning.

For 30-way 1-shot learning, we present the training and validation metrics in the figure below. These metrics include accuracy and loss curves, illustrating the performance of our model during the training and validation phases. The metrics show a clear convergence of the train and validation losses.

![Our training and validation metrics for 30-way 1-shot learning on the mini-ImageNet Dataset](https://hackmd-prod-images.s3-ap-northeast-1.amazonaws.com/uploads/upload_fff6bf7352d35c590eb0785a218eb9fa.png?AWSAccessKeyId=AKIA3XSAAW6AWSKNINWO&Expires=1714910006&Signature=km6wGIZA93wgmzutN1KNuOfao%2FI%3D)

Our results are summarised in the table below -

| Training    | Testing  | Our Accuracy | Original Accuracy |
|-------------|----------|----------------|------------|
| 20-way 5-shot     | 5-way 1-shot    | 0.4368|-|
|             | 5-way 5-shot    | 0.6442|0.6820|
| 30-way 1-shot     | 5-way 1-shot    | 0.4618|0.4942|
|             | 5-way 5-shot    | 0.6046| -|


# Running our code

To train the Protonet network, execute:

    $ python cli.py


The script takes the following command line options:

- `dataset`: Name of the dataset to run the experiment (miniimage or omniglot). Currently, only miniimage is supported.

- `data_path`: Path to the folder that contains the datasets. Should contain the unzipped data.

- `save_path`: Path where the experiment assets are saved.

- `num_epochs`: Number of epochs for training. Defaults to `100`.

- `num_episodes_train`: Number of episodes per epoch for training.

- `num_episodes_test`: Number of episodes to test on.

- `num_validation_steps`: Number of steps after which you conduct validation. Defaults to `100`.

- `learning_rate`: Learning rate for training. Default set based on original implementation per dataset.

- `lr_decay_step`: The number of steps after which the learning rate decays. Default set based on original implementation per dataset.

- `lr_decay_gamma`: Decay factor for the learning rate. Default set based on original implementation per dataset.

- `num_classes_train`: Number of classes to use in an episode while training. Default set based on original implementation per dataset.

- `num_support_train`: Number of support points to use in an episode while training. Default set based on original implementation per dataset.

- `num_classes_val`: Number of classes to use in an episode during validation.

- `num_support_val`: Number of support points to use in an episode during validation.

- `num_query_val`: Number of query points to use in an episode during validation.

- `conv_kernel_size`: Kernel size for the convolutional layers in the ProtoNet Encoder. Defaults to `3`

- `max_pool_kernel`: Kernel size for max pooling in the ProtoNet Encoder. Defaults to `3`

- `num_conv_layers`: The number of convolutional layers in the ProtoNet Encoder. Defaults to `4`

- `embedding_size`: An optional embedding size for the ProtoNet Encoder. This is to experiment with different embedding sizes.

- `distance_metric`: The distance metric to use. Defaults to `Euclidean`.

- `early_stopping_patience`: Patience for early stopping. Defaults to `3`

- `early_stopping_delta`: Delta for early stopping. Defaults to `0.05`


In order to run testing on a trained model, execute: 

    $ python cli_test.py

