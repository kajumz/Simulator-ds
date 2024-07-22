import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

img_path_ = "mug.jpg"
n_comp = 64
with Image.open(img_path_) as img:
    img = np.array(img)
me = img.mean(axis=0)
me = np.transpose(me)
print(me.shape)
#print(type(img[0][1][1]))
img = np.moveaxis(img, -1, 0)

#print(me)
print(img.shape)
img_c = img - me[:,:, np.newaxis]
print(img_c)

_, _, Vt = np.linalg.svd(img_c)
print(Vt.shape)
#Vt = Vt[:, :, :n_comp] # make Vt (3, 500, 64)
#print(Vt.shape)
#V = np.transpose(Vt, axes=(0, 2, 1)) # make (3, 64, 500)
#print(V.shape)
##Y = np.matmul(img_c, Vt)
#print(Y.shape)
#print(Vt)
#V = np.transpose(Vt)
#print(V.shape)
#X = np.matmul(Y, V)
#print(X.shape)
#X_me = X + me
#img_reconstructed = np.clip(X_me, 0, 255)
#print(X_me.shape)
#img_reconstructed = np.moveaxis(X_me, 0, -1).astype(np.uint8)
#print(img_reconstructed.shape)
#print(type(img_reconstructed[0][1][1]))
#a = Image.fromarray(img_reconstructed)
#a.save("reconstructed.jpg")


