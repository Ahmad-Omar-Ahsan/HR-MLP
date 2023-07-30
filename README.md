# Efficient Hyperbolic Perceptron for Image Classification
This repository contains the code for the paper "Efficient Hyperbolic Perceptron for Image Classification".

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Weights and biases](https://img.shields.io/badge/Weights_&_Biases-FFCC33?style=for-the-badge&logo=WeightsAndBiases&logoColor=black)


![HR-MLP](diagrams/HR-MLP.png)

## Setup environment
```
pip install -qr requirements.txt
```
## Set pythonpath
```
# From main dir
export PYTHONPATH="$PWD"
```
## Set WANDB API key

```
export WANDB_API_KEY = $your_key$
```

## Config file
Change hyperparameters in `sample_configs/base_config.yaml`

## Training

```
python3 pipeline.py --conf `path to config file`
```



