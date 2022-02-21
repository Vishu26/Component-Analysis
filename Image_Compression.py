import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob, os
import seaborn as sns
sns.set()
import numpy as np
from typing import List, Tuple


def load_image_path(PATH: str, ext: str='pgm') -> List[str]:

    a = []
    for x in os.walk(PATH):
        for y in glob.glob(os.path.join(x[0], F'*.{ext}')):
            a.append(y)
    return a


def disp_image(imagepaths: List[str], index: int) -> None:
    
    plt.figure()
    img = mpimg.imread(imagepaths[index])
    plt.grid(False)
    plt.imshow(img)
    plt.colorbar()


def get_image_size(imagepaths: List[str], index) -> Tuple[int]:

    img = mpimg.imread(imagepaths[index])
    return img.shape


def load_images(imagepaths: List[str]) -> np.array:

    m, n = get_image_size(imagepaths, 0)
    X = np.zeros((0, m*n), dtype=np.uint8)
    for path in enumerate(imagepaths):
        try:
            if get_image_size(imagepaths, path[0]) == (m, n):
                img = mpimg.imread(path[1])
                img = img.reshape(1, m*n)
                X = np.concatenate((X, img), axis=0)
        except:
            continue
    return X


def compute_eigens(X: np.array) -> Tuple[np.array]:

    assert len(X.shape) == 2
    CM = np.cov(X)
    eig, U = np.linalg.eig(CM)
    
    return eig, U


def plot_eigens(eig: np.array) -> None:

    plt.figure()
    plt.plot(eig)
    plt.title("Eigen Values")


def compress_images(X: np.array, U: np.array, N_COMPONENTS: int) -> np.array:

    A = U.T
    print(A.shape)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Image Compression using Principal Component Analysis')
    parser.add_argument('image_path', metavar='/usr/images/', type=str,
                        help='Root path of images')
    parser.add_argument('image_type', metavar='jpg', type=str,
                        help='Type of images to compress (jpg, png etc.)')
    parser.add_argument('n_components', metavar='5', type=int,
                        help='Number of principal components to use for compression')
    
    args = parser.parse_args()

    PATH = args.image_path
    IMAGE_TYPE = args.image_type
    N_COMPONENTS = args.n_components

    imagepaths = load_image_path(PATH, IMAGE_TYPE)
    
    disp_image(imagepaths, 0)
    
    X = load_images(imagepaths)
    eig, U = compute_eigens(X)
    plot_eigens(eig)

    compress_images(X, U, 5)

    plt.show()
