# imports cnn layer creation stuff
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# imports image reading stuff
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import PIL



# importing and reading the training datasets
train_datagen = ImageDataGenerator(rescale=1./255)

# set the training and validation/testing directories
training_dir = r'C:\Biorobotics\Bioinspired_artifical_inteligience\Coursework\datasets\training'
validation_dir =  r'C:\Biorobotics\Bioinspired_artifical_inteligience\Coursework\datasets\validation'

# load the training data
train_data = train_datagen.flow_from_directory(
    training_dir,
    target_size=(150, 150),
    batch_size=2, # small batch size as only being trained on a small amount of images
    class_mode='categorical'
)

# Load the validation data
validation_data = train_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=2,
        class_mode='categorical')

print("loaded training and testing data")



CNN = Sequential(name="CNN")

CNN.add(Conv2D(16, kernel_size=(3, 3),
               strides=(2, 2), padding="same",
               activation="relu", input_shape=(150, 150, 3)))

CNN.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                     padding="valid"))

# Add another pair of Conv2D and MaxPooling2D for more model depth,
# followed by the flatten and multiple dense layers

CNN.add(Conv2D(32, kernel_size=(3, 3),
               strides=(2, 2), padding="same",
               activation="relu"))

CNN.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                     padding="valid"))

CNN.add(Flatten())

CNN.add(Dense(64, activation='relu'))
CNN.add(Dense(32, activation='relu'))
CNN.add(Dense(2, activation='softmax'))

#summrary of all the layers
# CNN.summary()
print("Set up CNN")

# compile the model with the commonly used 'adam' optimizer
CNN.compile(optimizer= 'adam', # optimizer explains the formula for learning
              loss='sparse_categorical_crossentropy', # loss defines the weights for the loss function
              metrics=['accuracy']) # metrics are what is tested throughout the learning process

print("Compiled cnn")

# Train model on datasets
CNN.fit(train_data,
        batch_size=2,
        epochs=10,
        validation_data=validation_data)

# to address issues occurring like overfitting from using such a small dataset we can use some
# data augmentation techniques to increase effective size of our data set



print("entering evaluation")
# Evaluate the model on the validation data
scores = CNN.evaluate_generator(validation_data, steps=2, verbose=1)
print("Validation loss: ", scores[0])
print("Validation accuracy: ", scores[1])
print("finished eveluation")
