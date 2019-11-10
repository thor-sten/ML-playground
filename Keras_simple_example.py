from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# Data plotting
x = range(len(data))
width = 1
plt.bar(x, data, width)
# plt.show()

# Create model
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32)
predictions = model.predict(data)
