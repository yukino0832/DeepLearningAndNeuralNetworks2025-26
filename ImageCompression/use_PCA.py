from data.dataset import get_dataloader
import matplotlib.pyplot as plt
import numpy as np
from model.PCA import PCA
from easydict import EasyDict as edict

def visualize(images, labels):
    plt.figure(figsize=(10, 3))
    for (i, img) in enumerate(images):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        else:
            img = img.transpose(1, 2, 0)
        plt.subplot(1, 8, i+1)
        plt.imshow((img*255).astype(int), cmap='gray')
        plt.title('%s' % labels[i])
        plt.xticks([])
    plt.show()

cfg = edict({
    'channel': 1,
    'image_height': 32,
    'image_width': 32,
    'batch_size': 128,
    'embed_size': 30, # As per notebook
    'data_dir': 'data/MNIST',
})

_, val_ds = get_dataloader(
    data_dir=cfg.data_dir,
    batch_size=6,
    image_size=(cfg.image_height, cfg.image_width)
)
data_iter = iter(val_ds)
images, labels = next(data_iter)

oriImage = images 

images_flat = oriImage.reshape(6, -1).numpy()

_, reconImages = PCA(images_flat, cfg.embed_size)

reconImages = reconImages.view(np.ndarray).astype(np.float64).reshape((6, 1, cfg.image_height, cfg.image_width))

visualize(oriImage.numpy(), labels.numpy())
visualize(reconImages, labels.numpy())