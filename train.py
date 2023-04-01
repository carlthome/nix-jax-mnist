#!/usr/bin/env python
from jax import value_and_grad, random, numpy as np
from tqdm.auto import trange, tqdm
from tensorflow_datasets import load


def predict(x, W, b):
    logits = x @ W + b
    e_x = np.exp(logits - logits.max())
    activations = e_x / e_x.sum()
    return activations


def cross_entropy(y_true, y_pred):
    eps = 1e-9
    nll = (-y_true * np.log(y_pred + eps)).sum(axis=-1)
    return nll


def forward(W, b, x, y):
    y_pred = predict(x, W, b)
    return cross_entropy(y, y_pred).mean(axis=0)


def preprocess(examples):
    images = examples["image"]
    labels = examples["label"]

    num_examples = len(images)
    features = np.reshape(examples["image"], [num_examples, -1]) / 255.0
    targets = (
        np.zeros([num_examples, num_classes])
        .at[np.arange(num_examples), labels]
        .set(1.0)
    )
    return features, targets


if __name__ == "__main__":
    # Choose hyperparameters.
    learning_rate = 0.01
    batch_size = 32
    num_epochs = 10

    datasets, info = load("mnist", with_info=True, data_dir=".data")
    print(info)
    training_data = datasets["train"].shuffle(1024).batch(batch_size)
    test_data = datasets["test"].batch(batch_size)

    # Get input and output dimensionality for logistic regression.
    num_features = np.prod(np.asarray(info.features["image"].shape))
    num_classes = info.features["label"].num_classes

    # Create learnable parameters.
    key = random.PRNGKey(0)
    W = random.truncated_normal(
        key, lower=-1.0, upper=1.0, shape=[num_features, num_classes]
    )
    b = np.zeros([num_classes])

    # Train for a number of epochs.
    epochs = trange(num_epochs, desc="Epochs", unit="epoch")
    for epoch in epochs:

        # Train on training set.
        training = tqdm(
            training_data.as_numpy_iterator(),
            desc=f"Training",
            unit="minibatches",
            total=len(training_data),
            leave=False,
        )
        num_examples = 0
        total = 0.0
        for examples in training:
            x, y = preprocess(examples)

            loss, grads = value_and_grad(forward, argnums=[0, 1])(W, b, x, y)

            dldw, dldb = grads
            W = W + learning_rate * -dldw
            b = b + learning_rate * -dldb

            total += loss
            num_examples += len(x)
            training.set_postfix(loss=total / num_examples, refresh=False)
        average_training_loss = total / num_examples

        # Evaluate on test set.
        testing = tqdm(
            test_data.as_numpy_iterator(),
            desc=f"Evaluating",
            unit="minibatches",
            total=len(test_data),
            leave=False,
        )
        num_examples = 0
        total = 0
        for examples in testing:
            x, y = preprocess(examples)
            y_pred = predict(x, W, b)

            accuracy = np.count_nonzero(y.argmax(-1) == y_pred.argmax(-1))

            total += accuracy
            num_examples += len(x)
            testing.set_postfix(loss=total / num_examples, refresh=False)
        test_accuracy = total / num_examples

        epochs.set_postfix(
            epoch=epoch, average_loss=average_training_loss, accuracy=test_accuracy
        )
