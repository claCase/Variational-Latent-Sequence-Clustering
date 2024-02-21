import argparse
import numpy as np
import matplotlib.pyplot as plt
from src import models, utils
import tensorflow as tf
import os
from datetime import datetime

FIGURES = os.path.join(os.getcwd(), "figures")
LOG = os.path.join(os.getcwd(), "train_logs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--length", default=100, type=int)
    parser.add_argument("--sample_length", default=30, type=int)
    parser.add_argument("--samples", default=50, type=int)
    parser.add_argument("--batch_size", default=150, type=int)
    args = parser.parse_args()
    epochs = args.epochs
    length = args.length
    sample_length = args.sample_length
    samples = args.samples
    batch_size = args.batch_size

    now = datetime.now().strftime("%Y%m%d-%H%M%S")

    traj = utils.generate_sin(
        (0.1, 0.9, 2.0), noise=0.1, length=length, samples=samples
    )
    traj = tf.cast(
        tf.expand_dims(tf.transpose(tf.reshape(traj, (length, -1)), (1, 0)), -1),
        tf.float32,
    )

    mixvrnn = models.VariationalMixtureRNN(1, hidden_units=20, clusters=3)
    mixvrnn.compile("adam")
    _ = mixvrnn(traj)

    log_dir = os.path.join(LOG, now)
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir, update_freq=1, profile_batch="10, 15"
    )
    mixvrnn.fit(
        traj,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tb_callback,
        ],
    )

    save_path = os.path.join(FIGURES, now)
    os.makedirs(save_path)
    samples = mixvrnn.sample(traj, T=sample_length)
    x_samples = samples["x_sample"].numpy()
    np.save(os.path.join(save_path, "samples.npy"), x_samples)
    x_base = np.arange(length + sample_length)
    plt.figure(figsize=(16, 16))
    for i in range(1, 26):
        plt.subplot(5, 5, i)
        plt.plot(
            x_base[:-sample_length], x_samples[i * 5, :-sample_length, 0], color="blue"
        )
        plt.plot(
            x_base[-sample_length:], x_samples[i * 5, -sample_length:, 0], color="red"
        )
    plt.savefig(os.path.join(save_path, "prediction.png"))
