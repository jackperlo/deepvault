# !NOTE: constants MUST be in uppercase to be recognized by the codebase

# slice of images to evaluate the models on
N_IMAGES_SLICE = 2000

# quantized models top-1 accuracies on quantized Cifar-10 output file path
MODELS_ACCURACY_PATH = './datasets_data/_models_evaluation_stats.txt'

# cifar10 fp32 preprocessed data (for any fp32 model) 
FP32_TRAIN_SET_PREPROCESSED_PATH = './datasets_data/cifar10_fp32_data/cifar10_fp32_training_images_processed.npy'
FP32_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_fp32_data/cifar10_fp32_validation_images_processed.npy'

# cifar10 labels
TRAIN_LABELS_PATH = './datasets_data/cifar10_labels/cifar10_training_labels.npy'
VALIDATION_LABELS_PATH = './datasets_data/cifar10_labels/cifar10_validation_labels.npy'
VALIDATION_LABELS_2K_PATH = './datasets_data/cifar10_labels/cifar10_2k_validation_labels.npy'
VALIDATION_LABELS_500_PATH = './datasets_data/cifar10_labels/cifar10_500_validation_labels.npy'

#============================================================

# cifar10 x alexnet uint8 preprocessed validation images outputs path
ALEXNET_U8_MODEL_PATH = "./models_data/alexnet/alexnet_uint8_cifar10.tflite"
ALEXNET_U8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_alexnet/uint8/cifar10_u8_alexnet_validation'
ALEXNET_U8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_alexnet/uint8/cifar10_u8_alexnet_2k_validation'
ALEXNET_U8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_alexnet/uint8/cifar10_u8_alexnet_500_validation'

# cifar10 x alexnet int8 preprocessed validation images outputs path
ALEXNET_I8_MODEL_PATH = "./models_data/alexnet/alexnet_int8_cifar10.tflite"
ALEXNET_I8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_alexnet/int8/cifar10_i8_alexnet_validation'
ALEXNET_I8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_alexnet/int8/cifar10_i8_alexnet_2k_validation'
ALEXNET_I8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_alexnet/int8/cifar10_i8_alexnet_500_validation'

#============================================================

# cifar10 x resnet50 uint8 preprocessed validation images outputs path
RESNET50_U8_MODEL_PATH = "./models_data/resnet50/resnet50_uint8_cifar10.tflite"
RESNET50_U8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_resnet50/uint8/cifar10_u8_resnet50_validation'
RESNET50_U8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_resnet50/uint8/cifar10_u8_resnet50_2k_validation'
RESNET50_U8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_resnet50/uint8/cifar10_u8_resnet50_500_validation'

# cifar10 x resnet50 int8 preprocessed validation images outputs path
RESNET50_I8_MODEL_PATH = "./models_data/resnet50/resnet50_int8_cifar10.tflite"
RESNET50_I8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_resnet50/int8/cifar10_i8_resenet50_validation'
RESNET50_I8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_resnet50/int8/cifar10_i8_resnet50_2k_validation'
RESNET50_I8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_resnet50/int8/cifar10_i8_resnet50_500_validation'

#============================================================

# cifar10 x resnet152 uint8 preprocessed validation images outputs path
RESNET152_U8_MODEL_PATH = "./models_data/resnet152/resnet152_uint8_cifar10.tflite"
RESNET152_U8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_resnet152/uint8/cifar10_u8_resnet152_validation'
RESNET152_U8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_resnet152/uint8/cifar10_u8_resnet152_2k_validation'
RESNET152_U8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_resnet152/uint8/cifar10_u8_resnet152_500_validation'

# cifar10 x resnet152 int8 preprocessed validation images outputs path
RESNET152_I8_MODEL_PATH = "./models_data/resnet152/resnet152_int8_cifar10.tflite"
RESNET152_I8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_resnet152/int8/cifar10_i8_resenet152_validation'
RESNET152_I8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_resnet152/int8/cifar10_i8_resnet152_2k_validation'
RESNET152_I8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_resnet152/int8/cifar10_i8_resnet152_500_validation'

#============================================================

# cifar10 x vgg16 uint8 preprocessed validation images outputs path
VGG16_U8_MODEL_PATH = "./models_data/vgg16/vgg16_uint8_cifar10.tflite"
VGG16_U8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_vgg16/uint8/cifar10_u8_vgg16_validation'
VGG16_U8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_vgg16/uint8/cifar10_u8_vgg16_2k_validation'
VGG16_U8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_vgg16/uint8/cifar10_u8_vgg16_500_validation'

# cifar10 x vgg16 int8 preprocessed validation images outputs path
VGG16_I8_MODEL_PATH = "./models_data/vgg16/vgg16_int8_cifar10.tflite"
VGG16_I8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_vgg16/int8/cifar10_i8_vgg16_validation'
VGG16_I8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_vgg16/int8/cifar10_i8_vgg16_2k_validation'
VGG16_I8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_vgg16/int8/cifar10_i8_vgg16_500_validation'

#============================================================

# cifar10 x mobilenet uint8 preprocessed validation images outputs path
MOBILENET_U8_MODEL_PATH = "./models_data/mobilenet/mobilenet_uint8_cifar10.tflite"
MOBILENET_U8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_mobilenet/uint8/cifar10_u8_mobilenet_validation'
MOBILENET_U8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_mobilenet/uint8/cifar10_u8_mobilenet_2k_validation'
MOBILENET_U8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_mobilenet/uint8/cifar10_u8_mobilenet_500_validation'

# cifar10 x mobilenet int8 preprocessed validation images outputs path
MOBILENET_I8_MODEL_PATH = "./models_data/mobilenet/mobilenet_int8_cifar10.tflite"
MOBILENET_I8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_mobilenet/int8/cifar10_i8_mobilenet_validation'
MOBILENET_I8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_mobilenet/int8/cifar10_i8_mobilenet_2k_validation'
MOBILENET_I8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_mobilenet/int8/cifar10_i8_mobilenet_500_validation'

#============================================================

# cifar10 x densenet uint8 preprocessed validation images outputs path
MOBILENETV2_U8_MODEL_PATH = "./models_data/mobilenetV2/mobilenetV2_uint8_cifar10.tflite"
MOBILENETV2_U8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_mobilenetV2/uint8/cifar10_u8_mobilenetV2_validation'
MOBILENETV2_U8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_mobilenetV2/uint8/cifar10_u8_mobilenetV2_2k_validation'
MOBILENETV2_U8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_mobilenetV2/uint8/cifar10_u8_mobilenetV2_500_validation'

# cifar10 x densenet int8 preprocessed validation images outputs path
MOBILENETV2_I8_MODEL_PATH = "./models_data/mobilenetV2/mobilenetV2_int8_cifar10.tflite"
MOBILENETV2_I8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_mobilenetV2/int8/cifar10_i8_mobilenetV2_validation'
MOBILENETV2_I8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_mobilenetV2/int8/cifar10_i8_mobilenetV2_2k_validation'
MOBILENETV2_I8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_mobilenetV2/int8/cifar10_i8_mobilenetV2_500_validation'

#============================================================

# cifar10 x efficientnetB0 uint8 preprocessed validation images outputs path
EFFICIENTNETB0_U8_MODEL_PATH = "./models_data/efficientnetB0/efficientnetB0_uint8_cifar10.tflite"
EFFICIENTNETB0_U8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_efficientnetB0/uint8/cifar10_u8_efficientnetB0_validation'
EFFICIENTNETB0_U8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_efficientnetB0/uint8/cifar10_u8_efficientnetB0_2k_validation'
EFFICIENTNETB0_U8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_efficientnetB0/uint8/cifar10_u8_efficientnetB0_500_validation'

# cifar10 x efficientnetB0 int8 preprocessed validation images outputs path
EFFICIENTNETB0_I8_MODEL_PATH = "./models_data/efficientnetB0/efficientnetB0_int8_cifar10.tflite"
EFFICIENTNETB0_I8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_efficientnetB0/int8/cifar10_i8_efficientnetB0_validation'
EFFICIENTNETB0_I8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_efficientnetB0/int8/cifar10_i8_efficientnetB0_2k_validation'
EFFICIENTNETB0_I8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/cifar10_x_efficientnetB0/int8/cifar10_i8_efficientnetB0_500_validation'
