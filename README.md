# Training CartoonGAN

This is the repo for training CartoonGAN. The original paper of CartoonGAN can be found [here](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf).

## Requirments
* Make sure you have set up a python environment for tensorflow. More details of instructions can be found [here](https://ml5js.org/docs/training-setup).
* The version of the modules we use is listed in `requirements.txt`.

## Usage

### 1) Download this repository

### 2) Collect data
You should use the `download.py` to get the photos as well as cartoon images to be used during training.

```bash
python download.py
```

ATTENTION: Make sure you move the download images to the `dataset` folder which contains three sub-folders: `photo_imgs`, `cartoon_imgs` and `smooth_cartoon_imgs`. After that you have to run the `smooth_edge.py` to get the edge-smoothed images mentioned in the paper.

```bash
python smooth_edge.py
```

The dataset folder structure is as followed:
```
── dataset
      ├── photo_imgs
      |   ├── photo1.jpg
      |   ├── photo2.jpg
      |   └── ...
      |
      ├── cartoon_imgs
      |   ├── cartoon1.jpg
      |   ├── cartoon2.jpg
      |   └── ...
      |
      ├── smooth_cartoon_imgs
      |   ├── smooth1.jpg
      |   ├── smooth2.jpg
      |   └── ...
      |
      └── ...
```

### 3) Train

Run:

```bash
python main.py
```