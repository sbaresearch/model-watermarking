from PIL import Image
import numpy as np

def show_img_from_array(array, channels):
    if channels == 1:
        array = np.reshape(array, (array.shape[1], array.shape[2]))

        array = (array * 255).astype(np.float32)

    elif channels == 3:
        array = np.reshape(array, (array.shape[1], array.shape[2], array.shape[0]))

        array = (array * 255).astype(np.float32)

    img = Image.fromarray(array)
    img.show()
