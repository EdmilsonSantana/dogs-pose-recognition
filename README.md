# Dogs Pose Recognition

[![Roboflow Dataset](https://raw.githubusercontent.com/roboflow/notebooks/main/assets/badges/roboflow-dataset.svg)](https://universe.roboflow.com/emgs/dogs-pose-recognition)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/emgs/dogs-pose-recognition)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter_notebook-%23FF7A01.svg?logo=jupyter&logoColor=white)](notebooks/dogs-pose-recognition.ipynb)

<img src="./assets/banner.jpg" alt="Dog laying down" width="300" height="300">

## About the project

The main goal of this project is to utilize object detection to identify my dogs in three different positions: squatting, sitting, and lying down. As I don't have enough photos of them, I have utilized various data sources to create a comprehensive dataset:

- **Squatting**: I obtained the images from a public Instagram account called ["dogspoopinginprettyplaces"](https://www.instagram.com/dogspoopinginprettyplaces/?hl=en) using the [instaloader](https://instaloader.github.io/) library.
- **Sitting**: I manually selected similar dog images from the [Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) that were in a sitting position.
- **Lying Down**: While most of the images are from my own dogs, I also added a few more from the internet using the ["bing_image_downloader"](https://pypi.org/project/bing-image-downloader/) library.


## Training and Performance

|**Project Type**|**Model**|**Library**|**mAP - 0.5**|**CLS Loss (Validation)**|
|--|--|--|--|--|
|Object Detection|YOLOv5|PyTorch|0.88418|0.00802

### Reports

[![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-FFCC33?style=for-the-badge&logo=WeightsAndBiases&logoColor=black)](https://api.wandb.ai/links/emgs/hsbkxumh)

### Logs

- 🔍[Train Output](assets/train_output.log)