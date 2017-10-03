#
#
#

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

# Remove limits on numpy array print length
np.set_printoptions(threshold=np.inf)

# Fix random seed to allow reproducibility of bugs
seed = 9001
np.random.seed(seed)

# ==================================================
#               Data Preprocessing
# ==================================================

# Load images, labels in
images = np.load('data/images.npy')
labels = np.load('data/labels.npy')

# Convert image labels to "one-hot vectors"
one_hot_labels = keras.utils.to_categorical(labels)
#for l in one_hot_labels:
#    print(l)

# Contains the random, stratified sampling of data set
x_train, y_train = [], []
x_val, y_val = [], []
x_test, y_test = [], []

# Iterate through each image matrix, vectorize, and stratify it
index = 0
for i in images:
    ran_seed = random.random()
    
    # We convert to float32 to normalize pixel values as set of [0,1] later
    vectorized_image = np.reshape(i, -1).astype('float32')

    # Randomly stratify image into training, validation, or test sets
    if ran_seed < 0.6:
        x_train.append(vectorized_image)
        y_train.append(one_hot_labels[index])
    elif ran_seed < 0.75:
        x_val.append(vectorized_image)
        y_val.append(one_hot_labels[index])
    else:
        x_test.append(vectorized_image)
        y_test.append(one_hot_labels[index])
    index += 1

# Convert all sets to numpy arrays for use with Keras
x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Normalize pixel values as sets of 0 if the space contains no writing, or 1 if it does
#   This approach dramatically simplifies the problem for the computer
x_train /= 255
x_val /= 255
x_test /= 255

# Determine initial two layers of our model - the number of pixels and the number of classes
px_count = images.shape[1] * images.shape[2]
class_count = one_hot_labels.shape[1]

# ==================================================
#               Plotting sample images
# ==================================================

'''
plt.subplot(221)
plt.imshow(images[0], cmap=plt.get_cmap('gray'))
print(labels[0])
plt.subplot(222)
plt.imshow(images[1], cmap=plt.get_cmap('gray'))
print(labels[1])
plt.show()
'''


# ==================================================
#               Neural Net Model
# ==================================================

# Model Template
model = Sequential() # declare model
model.add(Dense(300, activation='relu', kernel_initializer='glorot_normal', input_shape=(px_count,))) # Simplified first layer
#model.add(Dense(class_count, activation='softmax', kernel_initializer='he_normal')) # Second layer to include number class

# ---------- Add more model layers here! ---------- #
#model.add(Dense(1028, activation='tanh', kernel_initializer='he_uniform'))
model.add(Dense(100, activation='selu', kernel_initializer='he_normal'))
#model.add(Dense(49, activation='selu', kernel_initializer='he_normal'))

model.add(Dense(class_count, kernel_initializer='he_normal', activation='softmax')) # Required final layer

# Compile Model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


# ==================================================
#               Train Model
# ==================================================

''' This fit gives mediocre accuracy over a medium-small period of time'''
history = model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs=100, batch_size=64, verbose=2)

''' This fit gives great accuracy over a medium-large period of time'''
#history = model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs=100, batch_size=12, verbose=2)

''' This fit gives exceptional accuracy over a large period of time'''
#history = model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs=125, batch_size=3, verbose=2)

''' This fit gives exceptional accuracy over an immense period of time'''
#history = model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs=150, batch_size=1, verbose=2)

# Report Results
print(history.history)
#scores = model.evaluate(x_val, y_val)
#print(scores)
prediction = model.predict(x_test, batch_size=64)


# ==================================================
#               Post Processing
# ==================================================

# Contains the list of projected (predicted) and actual results of the image
projection, actual = [], []

# Iterate through predictions, determining which value recieved the highest prediction and
#   marking that value in a list (denoting our prediction)
for p in prediction:
    
    # Reset max value, index, and max index each iteration
    maximum = float(0)
    index, max_index = 0, -1

    # Iterate through each set of preductions determining the highest predicted value
    for n in p:
        if float(n) > maximum:
            maximum = float(n)
            max_index = index
        index += 1

    # Mark our projection for this set in our list
    projection.append(max_index)

# Convert test set (actuals) back to standard numerical format
for t in y_test:
    actual.append(np.argmax(t))

# Generate Confusion Matrix
y_actual = pd.Series(actual, name='Actual')
y_predict = pd.Series(projection, name='Predicted')
confusion_matrix = pd.crosstab(y_actual, y_predict)

# Generate normalized confusion matrix
norm_confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)

# Generate full confusion matrix with totals
full_confusion_matrix = pd.crosstab(y_actual, y_predict, rownames=['Actual'], colnames=['Predicted'], margins=True)

print(norm_confusion_matrix)
print(full_confusion_matrix)

# Pretty plot pretty please
cmap = mpl.cm.get_cmap('Oranges')
plt.matshow(confusion_matrix, cmap=cmap)
plt.colorbar()
tick_marks = np.arange(len(confusion_matrix.columns))
plt.xticks(tick_marks, confusion_matrix.columns, rotation=45)
plt.yticks(tick_marks, confusion_matrix.index)

plt.ylabel(confusion_matrix.index.name)
plt.xlabel(confusion_matrix.columns.name)

#plt.show()
