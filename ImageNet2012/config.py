# slice of images to evaluate the models on
N_IMAGES_SLICE = 2000

# quantized models top-1 accuracies on quantized Imagenet2012 output file path
MODELS_ACCURACY_PATH = './datasets_data/_models_evaluation_stats.txt'

# imagenet2012 fp32 preprocessed data (for any fp32 model) 
FP32_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_fp32_data/imagenet2012_fp32_validation_images_processed.npy'

# imagenet2012 labels
VALIDATION_LABELS_PATH = './datasets_data/imagenet2012_labels/imagenet2012_validation_labels.npy'
VALIDATION_LABELS_2K_PATH = './datasets_data/imagenet2012_labels/imagenet2012_2k_validation_labels.npy' # first 2000 validation labels
VALIDATION_LABELS_500_PATH = './datasets_data/imagenet2012_labels/imagenet2012_500_validation_labels.npy' # first 500 validation labels

#============================================================

# imagenet2012 x ResNet50 uint8 preprocessed validation images outputs path
RESNET50_U8_MODEL_PATH = "./models_data/resnet50/resnet50_uint8_imagenet2012.tflite"
RESNET50_U8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_resnet50/uint8/imagenet2012_u8_resnet50_validation'
RESNET50_U8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_resnet50/uint8/imagenet2012_u8_resnet50_2k_validation'
RESNET50_U8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_resnet50/uint8/imagenet2012_u8_resnet50_500_validation'

# imagenet2012 x ResNet50 int8 preprocessed validation images outputs path
RESNET50_I8_MODEL_PATH = "./models_data/resnet50/resnet50_int8_imagenet2012.tflite"
RESNET50_I8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_resnet50/int8/imagenet2012_i8_resnet50_validation'
RESNET50_I8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_resnet50/int8/imagenet2012_i8_resnet50_2k_validation'
RESNET50_I8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_resnet50/int8/imagenet2012_i8_resnet50_500_validation'

#============================================================

# imagenet2012 x ResNet152 uint8 preprocessed validation images outputs path
RESNET152_U8_MODEL_PATH = "./models_data/resnet152/resnet152_uint8_imagenet2012.tflite"
RESNET152_U8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_resnet152/uint8/imagenet2012_u8_resnet152_validation'
RESNET152_U8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_resnet152/uint8/imagenet2012_u8_resnet152_2k_validation'
RESNET152_U8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_resnet152/uint8/imagenet2012_u8_resnet152_500_validation'

# imagenet2012 x ResNet152 int8 preprocessed validation images outputs path
RESNET152_I8_MODEL_PATH = "./models_data/resnet152/resnet152_int8_imagenet2012.tflite"
RESNET152_I8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_resnet152/int8/imagenet2012_i8_resnet152_validation'
RESNET152_I8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_resnet152/int8/imagenet2012_i8_resnet152_2k_validation'
RESNET152_I8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_resnet152/int8/imagenet2012_i8_resnet152_500_validation'

#============================================================

# imagenet2012 x VGG16 uint8 preprocessed validation images outputs path
VGG16_U8_MODEL_PATH = "./models_data/vgg16/vgg16_uint8_imagenet2012.tflite"
VGG16_U8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_vgg16/uint8/imagenet2012_u8_vgg16_validation'
VGG16_U8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_vgg16/uint8/imagenet2012_u8_vgg16_2k_validation'
VGG16_U8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_vgg16/uint8/imagenet2012_u8_vgg16_500_validation'

# imagenet2012 x VGG16 int8 preprocessed validation images outputs path
VGG16_I8_MODEL_PATH = "./models_data/vgg16/vgg16_int8_imagenet2012.tflite"
VGG16_I8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_vgg16/int8/imagenet2012_i8_vgg16_validation'
VGG16_I8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_vgg16/int8/imagenet2012_i8_vgg16_2k_validation'
VGG16_I8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_vgg16/int8/imagenet2012_i8_vgg16_500_validation'

#============================================================

# imagenet2012 x MobilenetV1 uint8 preprocessed validation images outputs path
MOBILENET_U8_MODEL_PATH = "./models_data/mobilenet/mobilenet_uint8_imagenet2012.tflite"
MOBILENET_U8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_mobilenet/uint8/imagenet2012_u8_mobilenet_validation'
MOBILENET_U8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_mobilenet/uint8/imagenet2012_u8_mobilenet_2k_validation'
MOBILENET_U8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_mobilenet/uint8/imagenet2012_u8_mobilenet_500_validation'

# imagenet2012 x MobilenetV1 int8 preprocessed validation images outputs path
MOBILENET_I8_MODEL_PATH = "./models_data/mobilenet/mobilenet_int8_imagenet2012.tflite"
MOBILENET_I8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_mobilenet/int8/imagenet2012_i8_mobilenet_validation'
MOBILENET_I8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_mobilenet/int8/imagenet2012_i8_mobilenet_2k_validation'
MOBILENET_I8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_mobilenet/int8/imagenet2012_i8_mobilenet_500_validation'

#============================================================

# imagenet2012 x MobileNetV2 uint8 preprocessed validation images outputs path
MOBILENETV2_U8_MODEL_PATH = "./models_data/mobilenetV2/mobilenetV2_uint8_imagenet2012.tflite"
MOBILENETV2_U8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_mobilenetV2/uint8/imagenet2012_u8_mobilenetV2_validation'
MOBILENETV2_U8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_mobilenetV2/uint8/imagenet2012_u8_mobilenetV2_2k_validation'
MOBILENETV2_U8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_mobilenetV2/uint8/imagenet2012_u8_mobilenetV2_500_validation'

# imagenet2012 x MobileNetV2 int8 preprocessed validation images outputs path
MOBILENETV2_I8_MODEL_PATH = "./models_data/mobilenetV2/mobilenetV2_int8_imagenet2012.tflite"
MOBILENETV2_I8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_mobilenetV2/int8/imagenet2012_i8_mobilenetV2_validation'
MOBILENETV2_I8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_mobilenetV2/int8/imagenet2012_i8_mobilenetV2_2k_validation'
MOBILENETV2_I8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_mobilenetV2/int8/imagenet2012_i8_mobilenetV2_500_validation'

#============================================================

# imagenet2012 x EfficientnetB0 uint8 preprocessed validation images outputs path
EFFICIENTNETB0_U8_MODEL_PATH = "./models_data/efficientnetB0/efficientnetB0_uint8_imagenet2012.tflite"
EFFICIENTNETB0_U8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_efficientnetB0/uint8/imagenet2012_u8_efficientnetB0_validation'
EFFICIENTNETB0_U8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_efficientnetB0/uint8/imagenet2012_u8_efficientnetB0_2k_validation'
EFFICIENTNETB0_U8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_efficientnetB0/uint8/imagenet2012_u8_efficientnetB0_500_validation'

# imagenet2012 x EfficientnetB0 int8 preprocessed validation images outputs path
EFFICIENTNETB0_I8_MODEL_PATH = "./models_data/efficientnetB0/efficientnetB0_int8_imagenet2012.tflite"
EFFICIENTNETB0_I8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_efficientnetB0/int8/imagenet2012_i8_efficientnetB0_validation'
EFFICIENTNETB0_I8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_efficientnetB0/int8/imagenet2012_i8_efficientnetB0_2k_validation'
EFFICIENTNETB0_I8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_efficientnetB0/int8/imagenet2012_i8_efficientnetB0_500_validation'

#============================================================

# imagenet2012 x ViT-b_16p uint8 preprocessed validation images outputs path
ViT_b_16p_224_U8_MODEL_PATH = "./models_data/ViT_b_16p_224/ViT_b_16p_224_uint8_imagenet2012.tflite"
ViT_b_16p_224_U8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_ViT_b_16p_224/uint8/imagenet2012_u8_ViT_b_16p_224_validation'
ViT_b_16p_224_U8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_ViT_b_16p_224/uint8/imagenet2012_u8_ViT_b_16p_224_2k_validation'
ViT_b_16p_224_U8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_ViT_b_16p_224/uint8/imagenet2012_u8_ViT_b_16p_224_500_validation'

# imagenet2012 x ViT-b_16p int8 preprocessed validation images outputs path
ViT_b_16p_224_I8_MODEL_PATH = "./models_data/ViT_b_16p/ViT_b_16p_224_int8_imagenet2012.tflite"
ViT_b_16p_224_I8_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_ViT_b_16p_224/int8/imagenet2012_i8_ViT_b_16p_224_validation'
ViT_b_16p_224_I8_2K_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_ViT_b_16p_224/int8/imagenet2012_i8_ViT_b_16p_224_2k_validation'
ViT_b_16p_224_I8_500_VALIDATION_SET_PREPROCESSED_PATH = './datasets_data/imagenet2012_x_ViT_b_16p_224/int8/imagenet2012_i8_ViT_b_16p_224_500_validation'
