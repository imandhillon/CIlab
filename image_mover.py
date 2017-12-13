import numpy as np
import scipy
from scipy import ndimage
from skimage import imsave, imread
import math
from PIL import Image

picture = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,1,1,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])

def mat_mover(img, trans=(0,0), rot=(0,0)):
    ''' img: an ndarray
        trans=(x,y)
        rot = (angle, direction)
    '''

    new_img = np.zeros(img.shape)

    dx = ndimage.sobel(img, 0)  # horizontal derivative
    dy = ndimage.sobel(img, 1)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    mag *= 255.0 / np.max(mag)  # normalize (Q&D)
    imsave('sobel.jpg', mag)

    # Translation
    for hgt in range(len(img)):
        for wid in range(len(img[hgt])):
            try:
                new_img[hgt+trans[1], wid+trans[0]] = img[hgt,wid]
            except:
                pass

    print(new_img)
    show_mat(new_img)
    

def show_mat(img, height_shown=3, width_shown=4, x=1, y=5):
    assert width_shown > 0 and height_shown > 0

    if width_shown+x > img.shape[1]:
        width_shown -= width_shown+x-img.shape[1]
    if y-height_shown < 0:
        height_shown -= y-height_shown-img.shape[0]

    mx=img[y-height_shown:y, x:width_shown+x]
    print(mx)


    '''
    for hgt in range(height_shown, y):
        for wid in range(x, width_shown+x):
            row.append(img[hgt][wid])
            
        mx.append(row)
        row = []
            


    for hgt in range(len(img)):
        #if hgt >= abs(y-height_shown) and hgt <= y and wid >= x and wid <= abs(x+width_shown):

        for wid in range(len(img[hgt])):
            if hgt >= abs(y-height_shown) and hgt <= y and wid >= x and wid <= abs(x+width_shown):
                #mx.append()

                print('&',img[hgt][wid], hgt, wid)'''
    

if __name__ == '__main__':
    show_mat(picture)
    mat_mover(picture, trans=(1,1))