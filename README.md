# deepvault
This repository brings together a personally curated collection of pre-trained models and datasets, designed to cut through the chaos of searching the internet for reliable resources. 

If you've ever felt lost in the jungle of scattered links and half-documented models—this is for you. Enjoy!

## How to use
The repository is divided per datasets.
1) Go to the dataset you would like to gather a model to be evaluated on

2.a.) If custom model is wanted: Search for the desired model notebook (e.g. Mnist/CNN/lenet5.ipynb) and train/customize it using the given .ipynb, eventually save it in the desired format (quantized or fp32)

2.b.) If the provided pre-trained model is wanted to be used, launch the download_<model>.sh found in the desired dataset and model folder:

```bash download_DESIRED-MODEL.sh```

or 

```chmod +x download_DESIRED-MODEL.sh```
```./download_DESIRED-MODEL.sh```