# InpaintGAN

## Introduction
This paper presents a novel approach to the image inpainting problem using a Generative Adversarial Network (GAN) architecture. In this paper, various edge generator modules have been proposed and examined to achieve the highest precision and recall in correlation to the lowest number of iterations possible.

The method proposed in this paper achieved a precision of 28%, recall of 25% and, feature matching loss of 25% while keeping the number of iterations at bare minimum of 175,000. In contrast, EdgeConnect [1] achieved precision of 27%, recall of 25% and, feature matching loss of 45%.
Our model can be useful for various image editing tasks such as image completion, object removal and image restoration, where fine details are crucial. Our approach addresses the limitation of current state-of-the-art models in producing fine detailed images and opens new possibilities for image inpainting applications.
The book is aimed at researchers, practitioners, and students in the fields of computer vision, image processing, and deep learning who are interested in learning about the latest advancements in image inpainting and how to improve current methods.

## Prerequisites
- Python 3
- PyTorch 1.13
- Eel 0.16
- scikit-image
- opencv 4.5.4.60
### Optional
- The development was utlizing PyCharm IDE by JetBrains

## Installation
- Cloning repo
```bash
git clone https://github.com/MoshPe/InpaintGAN.git
cd InpaintGAN
```
- Install Python dependencies
```bash
pip install -r requirements.txt
```
## Dataset
We use [HumanFaces](https://www.kaggle.com/datasets/ashwingupta3012/human-faces) datasets. To train the model on the data please download the dataset from Kaggle website with your registered user.
