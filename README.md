# Vision Transformers
Implementation of the SOTA vision transformers
The SOTA vision transformers are implemented in this repository from scratch in Tensorflow. The implemented models are:
* [ViT](https://arxiv.org/pdf/2010.11929.pdf)

In the future, this repository will be updated with most recent vision transformers.

## Install

### Clone Repository

Clone repo and install requirements.txt in a Python==3.10.6 environment, including Tensorflow==2.10.0.

```bash
git clone git@github.com:MrRiahi/Vision-Transformers.git
cd Vision-Transformers
```

### Virtual Environment
Python virtual environment will keep dependant Python packages from interfering with other Python projects on your
system.

```bash
python -m venv venv
source venv/bin/activate
``` 

### Requirements

Install python requirements.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Train 

Set your model name, number of epochs, dataset details in `src/config.py` and run the following command:

```bash
python train.py
```