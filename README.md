# ComfyUI Image Evaluation Node

This repository contains an extension to [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

This node allows you to evaluate the Clip Score between two images or a image and a text prompt using the CLIP model and DINO Score between two images using the DINO model.

## Installation
- Clone this repository into the `custom_nodes` folder of ComfyUI. 
```
cd ComfyUI-Image-Evaluation
```
- Install the requirements.
```
pip install -r requirements.txt
```
- Restart ComfyUI and the extension should be loaded.

## Features
![Alt text](images/clip_score.png)
- **Clip Text Score**: Evaluate the Clip Score between two images or a image and a text prompt using the CLIP model.
- **Clip Image Score**: Evaluate the Clip Score between a image and a target image using the CLIP model.
- **Dino Score**: Evaluate the DINO Score between two images using the DINO model.
- **Preload Clip model**
- **Multi text similarity score**
  ![F65C481D-7652-4D31-BC44-8236035FE3A2](https://github.com/user-attachments/assets/4adcdbff-be3c-47e1-9846-c8ab49a1ed00)

## Nodes

## Author
- Yujia Wu
- GitHub: [wu12023](https://github.com/wu12023)
