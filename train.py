#!/usr/bin/env python
from jax import value_and_grad, random, numpy as np
from tqdm.auto import trange, tqdm
from tensorflow_datasets import load


def predict(x, W, b):
    logits = x @ W + b
    e_x = np.exp(logits - logits.max())
    activations = e_x / e_x.sum()
    return activations


def loss(y_true, y_pred):
    eps = 1e-9
    nll = -y_true * np.log(y_pred + eps)
    return nll


def cost(W, b, x, y):
    y_pred = predict(x, W, b)
    return loss(y, y_pred).sum()


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
    epochs = 10

    datasets, info = load("mnist", with_info=True)
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

    for epoch in trange(epochs, desc="Training model.", unit="epoch"):

        # Train
        num_examples = 0
        total = 0.0
        for examples in tqdm(training_data.as_numpy_iterator(), desc=f"Training epoch {epoch}", unit='minibatches', position=1):
            x, y = preprocess(examples)

            l, grads = value_and_grad(cost, argnums=[0, 1])(W, b, x, y)

            dldw, dldb = grads
            W = W + learning_rate * -dldw
            b = b + learning_rate * -dldb

            num_examples += len(x)
            total += l

            #print(f"Loss: {total/num_examples}")

        # Evaluate
        num_examples = 0
        total = 0
        for examples in tqdm(test_data.as_numpy_iterator(), desc=f"Evaluating epoch {epoch}", unit="minibatches", position=1):
            x, y = preprocess(examples)
            y_pred = predict(x, W, b)

            total += np.count_nonzero(y.argmax(-1) == y_pred.argmax(-1))
            num_examples += len(x)

        average_training_loss = total / num_examples
        average_test_accuracy = total / num_examples

        print(f"Epoch: {epoch}, Training Loss: {average_training_loss}, Test accuracy: {average_test_accuracy*100}%")
