import matplotlib.pyplot as plt
import torch

def plot_generated_images(final_image, n_show=8):
    """
    Show up to `n_show` images from the batch in final_image.
    final_image: tensor of shape (B, 1, H, W)
    """
    final_image = final_image.detach().cpu()
    B = final_image.shape[0]
    n_show = min(n_show, B)

    # select first n_show images and remove channel dim
    imgs = final_image[:n_show].squeeze(1).numpy()  # shape: (n_show, H, W)
    imgs = (imgs + 1.0) / 2.0  # rescale to [0,1]

    fig, axes = plt.subplots(1, n_show, figsize=(n_show * 2, 2))
    if n_show == 1:
        axes = [axes]
    for ax, img in zip(axes, imgs):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')

    plt.tight_layout()
    plt.show()