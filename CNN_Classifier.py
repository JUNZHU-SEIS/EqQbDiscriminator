# Usage:	Instantiate a CNN model as an earthquake discriminator
# Author:	Jun ZHU
# Date:		AUG 23 2021
# Email:	Jun__Zhu@outlook.com


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
import keras.backend as K


def lr_schedule(epoch):
	"""learing rate schedule"""

	lr = 1e-3
	if epoch > 40:
		lr *= 0.5e-3
	elif epoch > 20:
		lr *= 1e-3
	elif epoch >10:
		lr *= 1e-1
	print('Learning rate: ', lr)
	return lr


def CNN(input_shape, dropout, lr=0.1, num_classes_output=2):
	"""set the architecture and configuration of the CNN model
		params:
			1. input shape (channels_last)
			2. dropout value
			3. **kwargs: learning rate; number of classes
		output:
			1. a CNN model, compiled with Adam
	"""

	K.clear_session()
	# basic configurations of conv layer
	kernel_size = (3, 1); conv_strides = (2, 1)
	pool_size = (3, 1); pool_strides = 1
	shallow_dropout = 0.1; l2_damping = 1e-4
	loss_function = {'binary':'binary_crossentropy',
			'categorical':'categorical_crossentropy'}
	# define a CNN model
	classifier = Sequential([
		#---------------------------conv layers--------------------------
		Conv2D(3, kernel_size=kernel_size, strides=conv_strides, activation='relu',
			input_shape=input_shape, kernel_regularizer=l2(l2_damping)),
		BatchNormalization(),
		MaxPooling2D(pool_size=pool_size, strides=pool_strides),
#		Dropout(shallow_dropout),

		Conv2D(8, kernel_size=kernel_size, strides=conv_strides,
			activation='relu', kernel_regularizer=l2(l2_damping)),
		BatchNormalization(),
		MaxPooling2D(pool_size=pool_size, strides=pool_strides),
#		Dropout(shallow_dropout),

		Conv2D(16, kernel_size=kernel_size, strides=conv_strides,
			activation='relu', kernel_regularizer=l2(l2_damping)),
		BatchNormalization(),
		MaxPooling2D(pool_size=pool_size, strides=pool_strides),
#		Dropout(shallow_dropout),

		Conv2D(32, kernel_size=kernel_size, strides=conv_strides,
			activation='relu', kernel_regularizer=l2(l2_damping)),
		BatchNormalization(),
		MaxPooling2D(pool_size=pool_size, strides=pool_strides),
#		Dropout(shallow_dropout),

		Conv2D(64, kernel_size=kernel_size, strides=conv_strides,
			activation='relu', kernel_regularizer=l2(l2_damping)),
		BatchNormalization(),
		MaxPooling2D(pool_size=pool_size, strides=pool_strides),
#		Dropout(shallow_dropout),

		Conv2D(128, kernel_size=kernel_size, strides=conv_strides,
			activation='relu', kernel_regularizer=l2(l2_damping)),
		BatchNormalization(),
		MaxPooling2D(pool_size=pool_size, strides=pool_strides),
#		Dropout(shallow_dropout),

		Conv2D(256, kernel_size=kernel_size, strides=conv_strides,
			activation='relu', kernel_regularizer=l2(l2_damping)),
		BatchNormalization(),
		MaxPooling2D(pool_size=pool_size, strides=pool_strides),

		#-----------------------------a big dropout before flatten-------
		Dropout(dropout),

		#-----------------------------flatten----------------------------
		Flatten(),

		#-----------------------------dense------------------------------
		Dense(num_classes_output, activation = 'softmax')],
		# name
		name = 'Classifier')
	classifier.compile(optimizer=Adam(lr=lr_schedule(0)),
						loss=loss_function['categorical'],
						metrics=['accuracy'])
	return classifier


if __name__=="__main__":
	from Config import HyperParams, Dir, Config
	hp = HyperParams(); conf = Config()
	dropout = hp.dropout
	lr = hp.lr
	num_classes_output = conf.num_classes
	input_shape = conf.input_shape
	model = CNN(input_shape, dropout, lr, num_classes_output)
	model.summary()
