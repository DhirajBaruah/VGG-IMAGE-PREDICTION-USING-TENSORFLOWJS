import tensorflowjs as tfjs
from tensorflow.keras.applications.vgg16 import VGG16
vgg16 = VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)
tfjs.converters.save_keras_model(vgg16,'./tfjs-models/VGG16')
