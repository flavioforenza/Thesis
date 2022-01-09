#import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import onnx
from onnx2keras import onnx_to_keras
import keras2onnx
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pickle
from PIL import Image

class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

#path_onnx_model = '/home/flavio/thesis/jetson_nano/jetson_benchmarks/benchmarks_pt2/ssd-mobilenet.onnx'
#teacher = onnx.load(path_onnx_model)
#teacher = onnx_to_keras(teacher, ['input_0'])
#teacher.save('./model/teacher.h5')

#import teacher
teacher = keras.models.load_model('./model/teacher.h5')

#view weights
for layer in teacher.layers:
    print(layer.name)


#keras.utils.plot_model(teacher)
#print(teacher.summary())

# Create the student
student = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
        layers.Flatten(),
        layers.Dense(10),
    ],
    name="student",
)

# Clone student for later comparison
student_scratch = keras.models.clone_model(student)

teacher.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
)

#config dataset
img_width, img_height = 300, 300 #width & height of input image
input_depth = 3 #RGB
imgs_folder = r'/home/flavio/thesis/jetson_nano/train-ssd/data/automotive/train/'
testing_data_dir = '/home/flavio/thesis/jetson_nano/train-ssd/data/automotive/test/'
#labels_path = '/home/flavio/thesis/jetson_nano/train-ssd/models/labels.txt'
labels = ['BACKGROUND', 'Bicycle', 'Building', 'Bus', 'Car', 'Motorcycle', 'Person', 'Traffic light', 'Traffic sign', 'Train', 'Truck']
epochs = 2 #number of training epoch
batch_size = 5 #training batch size

def create_dataset(img_folder):   
    img_data_array=[]
    for file in os.listdir(img_folder):
        image_path= os.path.join(img_folder, file)
        image= np.array(Image.open(image_path))
        image= np.resize(image, (img_height, img_width, 3))
        image = image.astype('float32')
        image /= 255 
        img_data_array.append(image)
    return img_data_array

img_data =create_dataset(imgs_folder)

target_dict={k: v for v, k in enumerate(np.unique(labels))}
print(target_dict)

target_val=  [target_dict[labels[i]] for i in range(len(labels))]
print("TARGET: ", target_val)

history = distiller.fit(x=np.array(img_data, np.float32), y=np.array(list(map(int,target_val)), np.float32), epochs=3)


#addestrare il distiller prima di esportare il tutto
#onnx_model = keras2onnx.convert_keras(distiller, distiller.name)
#print(onnx_model)