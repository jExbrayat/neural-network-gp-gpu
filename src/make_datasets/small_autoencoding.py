import numpy as np
import matplotlib.pyplot as plt

data_dim = 20
xx = np.linspace(-1, 1, data_dim)
gaussian_noise = + np.random.randn(data_dim) * 0.1
yy = np.sin(xx*2*np.pi) + gaussian_noise 
yy = (yy - np.min(yy)) / (np.max(yy) - np.min(yy))
plt.plot(xx, yy)
plt.show()

# Create dataset
dataset = np.empty((0, data_dim))
for i in range(300):
    gaussian_noise = + np.random.randn(data_dim) * 0.1
    noisy_sin = np.sin(xx*2*np.pi) + gaussian_noise
    noisy_sin = (noisy_sin - np.min(noisy_sin)) / (np.max(noisy_sin) - np.min(noisy_sin))
    noisy_sin = noisy_sin.reshape((-1, 20))
    dataset = np.concatenate([dataset, noisy_sin], axis=0)

np.savetxt("datasets/noisy_sinus.csv", dataset, delimiter=",")






