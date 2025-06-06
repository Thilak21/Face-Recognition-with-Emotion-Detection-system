from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

#image size

img_width, img_height = 48,48

#Create ImageDataGenerators

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'datasets/train',
    target_size = (img_width,img_height),
    color_mode = 'grayscale',
    batch_size = 64,
    class_mode = 'categorical')

validation_generator = test_datagen.flow_from_directory(
    'datasets/test',
    target_size = (img_width,img_height),
    color_mode = 'grayscale',
    batch_size = 64,
    class_mode = 'categorical')

#Build CNN model

model = Sequential()

model.add(Conv2D(32,(3,3), activation='relu',input_shape=(img_width,img_height,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0,25))

model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0,25))

model.add(Conv2D(128,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0,25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0,5))
model.add(Dense(train_generator.num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

#Train the model

model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // train_generator.batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size)

model.save('emotion_model.hdf5')
print('Model saved as emotion_model_folder.hdf5')

                     































    
