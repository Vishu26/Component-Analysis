import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob, os
import seaborn as sns
sns.set()
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from prettytable import PrettyTable
import warnings


def load_image_path(PATH: str, ext: str='pgm') -> List[str]:

    a = []
    for x in os.walk(PATH):
        for y in glob.glob(os.path.join(x[0], F'*.{ext}')):
            a.append(y)
    return a


def disp_image(imagepaths: List[str], index: int) -> None:
    
    plt.figure()
    img = mpimg.imread(imagepaths[index]) / 255
    plt.grid(False)
    plt.imshow(img)
    plt.colorbar()
    plt.title("Original Image")


def get_image_size(imagepaths: List[str], index) -> Tuple[int]:

    img = mpimg.imread(imagepaths[index])
    return img.shape


def load_images(imagepaths: List[str]) -> np.array:

    m, n = get_image_size(imagepaths, 0)
    X = np.zeros((0, m*n), dtype=np.float32)
    for path in enumerate(imagepaths):
        try:
            if get_image_size(imagepaths, path[0]) == (m, n):
                img = mpimg.imread(path[1])
                img = img.reshape(1, m*n)
                X = np.concatenate((X, img), axis=0)
        except:
            continue
    X = X / 255

    return X


def compute_eigens(X: np.array) -> Tuple[np.array]:

    assert len(X.shape) == 2
    CM = np.cov(X.T)
    eig, U = np.linalg.eig(CM)
    
    return eig, U


def plot_eigens(eig: np.array) -> None:

    plt.figure()
    plt.plot(eig)
    plt.title("Eigen Values")


def compress_images(X: np.array, U: np.array, N_COMPONENTS: int) -> np.array:

    A = U.T
    Y = A[:N_COMPONENTS, :].dot(X.T)

    return Y


def reconstruct_images(Y: np.array, U: np.array, N_COMPONENTS: int) -> np.array:

    A = U[:, :N_COMPONENTS]
    X = A.dot(Y)
    X = X.T
    X = X.astype(np.float32)

    return X


def disp_recon_image(X: np.array, nrow: int, ncol: int, index: int) -> None:

    img = X[index].reshape(m, n)
    plt.figure()
    plt.grid(False)
    plt.imshow(img)
    plt.colorbar()
    plt.title("Reconstructed Image")


def calculate_loss(X: np.array, Xr: np.array) -> Tuple[float]:
    
    SE = (X - Xr)**2
    MSE = np.mean(SE)
    RMSE = np.sqrt(MSE)

    return RMSE, MSE


def plot_losses(X: np.array, U: np.array, N_COMPONENTS: int) -> None:

    mse_losses = []
    rmse_losses = []
    n_features = U.shape[1]
    assert N_COMPONENTS <= n_features
    
    for i in tqdm(range(1, N_COMPONENTS+1)):
        Y = compress_images(X, U, i)
        Xr = reconstruct_images(Y, U, i)
        rmse, mse = calculate_loss(X, Xr)
        mse_losses.append(mse)
        rmse_losses.append(rmse)
    
    plt.figure()
    plt.plot(range(1, N_COMPONENTS+1), mse_losses)
    plt.title("Mean Squared Error vs No of Principal Components Selected")
    plt.xlabel("No of Principal Components")
    plt.ylabel("Mean Squared Error")

    print()
    table = PrettyTable()

    table.add_column("No. of PC", range(1, N_COMPONENTS+1))
    table.add_column("MSE", mse_losses)
    table.add_column("RMSE", rmse_losses)

    print(table)


if __name__ == '__main__':

    warnings.filterwarnings("ignore")

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
    
    m, n = get_image_size(imagepaths, 0)

    disp_image(imagepaths, 0)
    
    X = load_images(imagepaths)
    eig, U = compute_eigens(X)
    plot_eigens(eig)
    
    Y = compress_images(X, U, 5)

    Xr = reconstruct_images(Y, U, 5)

    disp_recon_image(Xr, m, n, 0)

    rmse, mse = calculate_loss(X, Xr)

    print()
    print(F"RMSE: {rmse}, MSE: {mse}")
    print()

    plot_losses(X, U, 10)

    plt.show()
