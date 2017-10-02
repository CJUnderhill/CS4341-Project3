import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

# Remove limits on numpy array print length
np.set_printoptions(threshold=np.inf)

# Load images in
images = np.load('data/images.npy')

# Load labels in
labels = np.load('data/labels.npy')
NUM_CLASSES = 10

# Preprocess images
vectorized_images = []
for i in images:
    vectorized_images.append(np.reshape(i, -1))
#print('Image 0:')
#print(images[0])
#print(vectorized_images[0])
#print('Image 1:')
#print(images[1])
#print(vectorized_images[1])


# Convert image labels to "one-hot vectors" of length ten
one_hot_labels = keras.utils.to_categorical(labels, NUM_CLASSES)
#for l in one_hot_labels:
#    print(l)



# Model Template

#model = Sequential() # declare model
#model.add(Dense(10, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
#model.add(Activation('relu'))
#
#
#
# Fill in Model Here
#
#
#model.add(Dense(10, kernel_initializer='he_normal')) # last layer
#model.add(Activation('softmax'))


# Compile Model
#model.compile(optimizer='sgd',
#              loss='categorical_crossentropy', 
#              metrics=['accuracy'])

# Train Model
#history = model.fit(x_train, y_train, 
#                    validation_data = (x_val, y_val), 
#                    epochs=10, 
#                    batch_size=512)


# Report Results

#print(history.history)
#model.predict()