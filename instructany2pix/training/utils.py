import wandb
from matplotlib import pyplot as plt
import torch
def wandb_dump_images(imgs, name="vis", keys=None, **kwargs):
    """
    x: H X W X C
    y: H X W X C
    """
    if wandb.run is not None:
        n_imgs = len(imgs)
        fig, axes = plt.subplots(1, n_imgs, figsize=(5 * n_imgs, 5))
        for idx, img in enumerate(imgs):
            if torch.is_tensor(img):
                img = img.detach().cpu().float().numpy()
            axes[idx].imshow(img)
            if keys:
                axes[idx].title.set_text(keys[idx])
        fig.tight_layout()
        wandb.log({name: wandb.Image(fig), **kwargs})
        plt.close(fig)