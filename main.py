import numpy as np 
import matplotlib.pyplot as plt 
from src import models, utils 
import tensorflow as tf 

if __name__ == "__main__":
    T = 100
    traj = utils.generate_sin((0.1, 0.9, 2.),noise=0.1, length=T)
    traj = tf.cast(tf.expand_dims(tf.transpose(tf.reshape(traj, (100, -1)), (1, 0)), -1),tf.float32)

    mixvrnn = models.VariationalMixtureRNN(1, hidden_units=20, clusters=3)
    mixvrnn.compile("adam")
    _ = mixvrnn(traj)

    epochs = 1500
    pbar = tf.keras.utils.ProgBar(epochs)

    for i in range(epochs):
        h = mixvrnn.train_step(traj)
        pbar.update(i, values=h.items(), finalize=False)
    pbar.update(i+1, values=h.items(), finalize=True)

    samples = mixvrnn.sample(traj)
    x_samples = samples["x_sample"]
    
    i = 0
    x_base = np.linspace(0, 1, 115)
    plt.plot(x_base[:-15], x_samples[i, :-15, 0], color="blue")
    plt.plot(x_base[-15:], x_samples[i, 15:, 0], color="red")