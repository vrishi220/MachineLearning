from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation

(X_train, y_train), (X_test, y_test) = mnist.load_data()
output_size, classes_size, num_epoch, batch_size, input_size  = 10, 10, 20, 50, 28**2
X_train, X_test = X_train.reshape(X_train.shape[0], input_size).astype('float32')/255, X_test.reshape(X_test.shape[0], input_size).astype('float32')/255
Y_train, Y_test = np_utils.to_categorical(y_train, classes_size), np_utils.to_categorical(y_test, classes_size)
nn = Sequential()
nn.add(Dense(output_size, input_dim = input_size, activation = 'softmax'))
nn.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
nn.fit(X_train, Y_train, batch_size = batch_size, nb_epoch = num_epoch, verbose = 2, validation_data = (X_test, Y_test))
print('Accuracy:', nn.evaluate(X_test, Y_test, verbose = 2)[1])
