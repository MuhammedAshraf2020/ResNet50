from keras.layers import Conv2D , AveragePooling2D , ZeroPadding2D , MaxPooling2D
from keras.layers import BatchNormalization , Activation
from keras.layers import Flatten , Dense , Add , Input
from keras.initializers import glorot_uniform
from keras.models import Model


def identity_block(X , f , filters , stage , block):

	#Defining the basis
	conv_name_base = "res" + str(stage) + block + "_branch"
	bn_name_base   = "bn"  + str(stage) + block + "_branch"

	#Retrieve Filters
	F1 , F2 , F3 = filters

	X_shortcut = X
	
	#First component of main path
	X = Conv2D(filters = F1 , kernel_size = (1 , 1) , strides = (1 , 1) ,
				 padding = "valid" , name = conv_name_base + "2a" , kernel_initializer = glorot_uniform(seed = 0))(X)
	X = BatchNormalization(axis = 3 , name = bn_name_base  + "2a")(X)
	X = Activation("relu")(X)

	#Second component of main path
	X = Conv2D(filters = F2 , kernel_size = (f , f) , strides = (1 , 1) ,
				 padding = "same" , name = conv_name_base + "2b" , kernel_initializer = glorot_uniform(seed = 0))(X)
	X = BatchNormalization(axis = 3 , name = bn_name_base  + "2b")(X)
	X = Activation("relu")(X)

	# Third component of main path
	X = Conv2D(filters = F3 , kernel_size = (1 , 1) , strides = (1 , 1) ,
				 padding = "valid" , name = conv_name_base + "2c" , kernel_initializer = glorot_uniform(seed = 0))(X)
	X = BatchNormalization(axis = 3 , name = bn_name_base  + "2c")(X)
	#Final Step
	X = Add()([X , X_shortcut])
	X = Activation("relu")(X)

	return X

def convolutional_block(X , f , filters , stage , block , s):

	#Defining the basis
	conv_name_base = "res" + str(stage) + block + "_branch"
	bn_name_base   = "bn"  + str(stage) + block + "_branch"

	#Retrieve Filters
	F1 , F2 , F3 = filters

	X_shortcut = X
	
	#First component of main path
	X = Conv2D(filters = F1 , kernel_size = (1 , 1) , strides = (1 , 1) ,
				 padding = "valid" , name = conv_name_base + "2a" , kernel_initializer = glorot_uniform(seed = 0))(X)
	X = BatchNormalization(axis = 3 , name = bn_name_base  + "2a")(X)
	X = Activation("relu")(X)

	#Second component of main path
	X = Conv2D(filters = F2 , kernel_size = (f , f) , strides = (1 , 1) ,
				 padding = "same" , name = conv_name_base + "2b" , kernel_initializer = glorot_uniform(seed = 0))(X)
	X = BatchNormalization(axis = 3 , name = bn_name_base  + "2b")(X)
	X = Activation("relu")(X)

	# Third component of main path
	X = Conv2D(filters = F3 , kernel_size = (1 , 1) , strides = (1 , 1) ,
				 padding = "valid" , name = conv_name_base + "2c" , kernel_initializer = glorot_uniform(seed = 0))(X)
	X = BatchNormalization(axis = 3 , name = bn_name_base  + "2c")(X)
	
	#Final Step
	X_shortcut = Conv2D(filters = F3 , kernel_size = (1 , 1) , strides = (s , s) , padding = "valid" ,
						name = conv_name_base + "1" , kernel_initializer = glorot_uniform(seed = 0))(X_shortcut)
	X = BatchNormalization(axis = 3 , name = bn_name_base  + "1")(X_shortcut)

	X = Add()([X , X_shortcut])
	X = Activation("relu")(X)

	return X