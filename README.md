# Training CartoonGAN

This is the repo for training the CartoonGAN. The original paper of CartoonGAN can be found [here](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf).

## Requirments
* Make sure you have set up a python environment for tensorflow. More details of instructions can be found [here](https://ml5js.org/docs/training-setup).
* The version of the modules we use is listed in `requirements.txt`.

## Usage

### 1) Download this repository
Start by [downloading](https://github.com/Derek-Wds/training_CartoonGAN.git) this repo or clone this repository:
```bash
git clone https://github.com/Derek-Wds/training_CartoonGAN.git
cd training_CartoonGAN
```

### 2) Collect data
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
You could use your own dataset, but pay
**ATTENTION**: Make sure you move the download images to the `dataset` folder which contains three sub-folders: `photo_imgs`, `cartoon_imgs` and `smooth_cartoon_imgs`. After that you have to run the `smooth_edge.py` to get the Gaussian edge-smoothed images.

```bash
python smooth_edge.py
```

After making the dataset, now should use folllowing command to preprocess the images in order to feed them into the model. This step will create corresponding folders for the numpy files.

```bash
python preprocess.py
```

If you do not have available data on hand, you could use the `download.py` to get the photos as well as cartoon images to be used during training.

```bash
python download.py
```


### 3) Train

Run the training script with default arguments:

```bash
python main.py
```

Or you could specify your preferred hyperparameters settings:
```
python main.py --batch_size=32 \
--epochs=500 \
--gpu_num=2 \
--image_channels=3 \
--image_size=256 \
--init_epoch=30 \
--lr=0.0002
```

Or you could run bash script `run.sh`:
```bash
bash run.sh
```

If you want to resume the training, you could just use `load_weights` to load the saved weights of the modals (`generator`, `discriminator`, and `train_generator`) and keep training.

### 4) Visulization
We provide access to the visualization of loss and generated images. You could use following command to do this:

```
cd logs
tensorboard --logdir logs --port 9090
```

Then you could have access to tensorboard at `localhost:9090`.

### 5) Use it

That's it and have fun with CartoonGAN in ml5js!
