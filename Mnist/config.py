# quantized models top-1 accuracies on quantized mnist output file path
MODELS_ACCURACY_PATH = './datasets_data/_models_evaluation_stats.txt'

# mnist fp32 preprocessed data (for any fp32 model) 
FP32_TRAIN_SET_PREPROCESSED_PATH = './datasets_data/mnist_fp32_data/mnist_fp32_train.npy'
FP32_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/mnist_fp32_data/mnist_fp32_validation.npy'

# mnist labels output file paths
TRAIN_LABELS_PATH = './datasets_data/mnist_labels/mnist_training_labels.npy'
VALIDATION_LABELS_PATH = './datasets_data/mnist_labels/mnist_validation_labels.npy'
VALIDATION_LABELS_2K_PATH = './datasets_data/mnist_labels/mnist_2k_validation_labels.npy' # first 2000 validation labels
VALIDATION_LABELS_500_PATH = './datasets_data/mnist_labels/mnist_500_validation_labels.npy' # first 500 validation labels

# mnist x Lenet5 uint8 preprocessed train and validation images outputs path
LENET5_U8_MODEL_PATH = "./models_data/lenet5/lenet5_uint8_mnist.tflite"
LENET5_U8_TRAIN_SET_PREPROCESSED_PATH = './datasets_data/mnist_x_lenet5/uint8/mnist_u8_lenet5_training'
LENET5_U8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/mnist_x_lenet5/uint8/mnist_u8_lenet5_validation'
LENET5_U8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/mnist_x_lenet5/uint8/mnist_u8_lenet5_2k_validation'
LENET5_U8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/mnist_x_lenet5/uint8/mnist_u8_lenet5_500_validation'

# mnist x Lenet5 int8 preprocessed train and validation images outputs path
LENET5_I8_MODEL_PATH = "./models_data/lenet5/lenet5_int8_mnist.tflite"
LENET5_I8_TRAIN_SET_PREPROCESSED_PATH = './datasets_data/mnist_x_lenet5/int8/mnist_i8_lenet5_training'
LENET5_I8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/mnist_x_lenet5/int8/mnist_i8_lenet5_validation'
LENET5_I8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/mnist_x_lenet5/int8/mnist_i8_lenet5_2k_validation'
LENET5_I8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/mnist_x_lenet5/int8/mnist_i8_lenet5_500_validation'
