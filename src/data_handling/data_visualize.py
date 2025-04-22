import matplotlib.pyplot as plt
import numpy as np

def visualize_image_depth(img: np.ndarray, depth: np.ndarray):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img)
    axes[1].imshow(depth)