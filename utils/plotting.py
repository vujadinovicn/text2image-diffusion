import matplotlib.pyplot as plt
import torch

def plot_sample_generated_images(final_image, generated_images):
    final_image = final_image.squeeze().cpu().numpy()
    final_image = (final_image + 1) / 2  
    plt.imshow(final_image, cmap='gray')

    # plot_images = generated_images[-10:]
    # take equally spaced 10 images from generated_images
    plot_images = []
    num_images = len(generated_images)
    indices = torch.linspace(0, num_images - 1, steps=10).long()
    for idx in indices:
        plot_images.append(generated_images[idx])

    fig, axes = plt.subplots(1, len(plot_images), figsize=(15, 3))
    for ax, img in zip(axes, plot_images):
        img = img.squeeze().cpu().numpy()
        img = (img + 1) / 2  # Rescale to [0, 1]
        # img = img.clip(0, 1)
        ax.imshow(img)
        ax.axis('off')
    plt.show()