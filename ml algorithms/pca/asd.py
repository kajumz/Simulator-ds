import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage


class ImagePCA:
    """
    This class compresses an image using PCA and reconstructs it upon calling.

    Attributes
    ----------
    img_path_ : str
        Stores the path of the image file.
        Load the jpg image of shape (H, W, 3)
    n_components_ : int
        Stores the number of principal components used for compression.
    mean_ : ndarray of the shape (W, 3)
        The mean value of the original image used for centering.
    Y_ : ndarray of the shape (n_components_, H, W, 3)
        The compressed image data after applying PCA.
    Vt_ : ndarray of the shape (3, n_components_, W)
        The top n principal components from the SVD.
    """

    def __init__(self, img_path: str, n_components: int) -> None:
        self.img_path_ = img_path
        self.n_components_ = n_components

        self._compress()

    def _compress(self) -> None:
        """
        Compress the image using Singular Value Decomposition (SVD).

        The image is first converted to an array and centered. Then SVD is applied
        to compress the image data. This method sets the Y_ and Vt_ attributes.
        """
        with Image.open(self.img_path_) as img:
            img = np.array(img)

        self.mean_ = img.mean(axis=0) #(500, 3)
        #self.mean_ = np.transpose(self.mean_)
        #img = np.moveaxis(img, -1, 0)  # Change shape from (H, W, 3) to (3, H, W)



        img_centered = img - self.mean_[np.newaxis, :, :]
        img_centered = np.moveaxis(img_centered, -1, 0)
        _, _, Vt = np.linalg.svd(img_centered)

        self.Vt_ = Vt[:, :self.n_components_, :] # (3, n_Comp, 500)

        V = np.transpose(self.Vt_, axes=(0, 2, 1)) # (3, 500, n_comp)
        #print(V.shape)
        self.Y_ = np.matmul(img_centered, V) # (3, 500,500) * (3, 500, n_comp)
        #print(self.Y_.shape)

    def __call__(self) -> PILImage:
        """
        Reconstruct and return the compressed image.

        When the instance is called, it reconstructs the image from the compressed
        data and returns it as a PIL Image object.

        Returns
        -------
        PILImage
            The reconstructed image after PCA compression.
        """

        img_reconstructed = np.matmul(self.Y_, self.Vt_) # (3, 500, n_comp) * (3, n_comp, 500)
        img_reconstructed = np.moveaxis(img_reconstructed, 0, -1)
        img_reconstructed = img_reconstructed + self.mean_[np.newaxis, :, :]
        img_reconstructed = np.clip(img_reconstructed, 0, 255)
        img_reconstructed = img_reconstructed.astype(np.uint8)
        #img_reconstructed = np.moveaxis(img_reconstructed, 0, -1).astype(np.uint8) # (500, 500, 3)

        return Image.fromarray(img_reconstructed)


if __name__ == "__main__":
    img_path = "mug.jpg"
    n_components = 64

    img_pca = ImagePCA(img_path, n_components)
    img = img_pca()

    img.save("reconstructed.jpg")
