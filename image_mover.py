import numpy as np
import scipy
from scipy import ndimage
from skimage.io import imsave, imread
import math
from PIL import Image
import cv2


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))




def mat_mover(img, trans=(0,0), rot_angle=0):
    ''' img: an ndarray
        trans=(x,y)
        rot = (angle, direction)
    '''

    trans_img = np.zeros(img.shape)

    img = ndimage.gaussian_filter(img, 8)

    dx = ndimage.sobel(img, 0)  # horizontal derivative
    dy = ndimage.sobel(img, 1)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    #mag *= 255.0 / np.max(mag)  # normalize (Q&D)      This line breaks it. Not sure if needed
    print('dx:', dx, '\ndy:', dy, '\nmag:', type(mag))

    cv2.imwrite('sobel.jpg', mag)


    # Translation
    for hgt in range(len(img)):
        for wid in range(len(img[hgt])):
            try:
                trans_img[hgt+trans[1], wid+trans[0]] = img[hgt,wid]
            except:
                pass

    print(trans_img)
    show_mat(trans_img)

    # Rotation                                                      #################
    #rows, cols = img.shape
    #rotation_m = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    #new_img = cv2.warpAffine(img,rotation_m,(cols,rows))

    rotated = rotate_bound(trans_img, rot_angle)
    cv2.imshow("Rotated", rotated)

    print(rotated)
    show_mat(rotated)

    

def show_mat(img, height_shown=3, width_shown=4, x=1, y=5):
    assert width_shown > 0 and height_shown > 0

    if width_shown+x > img.shape[1]:
        width_shown -= width_shown+x-img.shape[1]
    if y-height_shown < 0:
        height_shown -= y-height_shown-img.shape[0]

    mx=img[y-height_shown:y, x:width_shown+x]
    print(mx)
    

if __name__ == '__main__':

    '''picture = np.array([[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,1,1,1,1],
                        [0,0,1,1,0,1],
                        [0,0,1,1,1,1],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]])
    '''
    picture = imread('square.jpg')

    print(picture, picture.shape)
    cv2.imshow('window',picture)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
                        
    show_mat(picture)
    mat_mover(picture, trans=(1,1), rot_angle=90)