import numpy as np
import cv2
from enum import Enum



class Noises(Enum):
    NOIS_GAUSS="gauss"
    NOIS_SALT_AND_PEPPER="s&p"
    NOIS_POISSON="poisson"
    NOIS_SPECKLE="speckle"


def make_hsv_equalized(image,hsv=True):
    """ make histogram equalization"""
    
    im=np.swapaxes(image.numpy(),0,2)
    im = 255 * im # Now scale by 255
    im = im.astype(np.uint8)
    R, G, B = cv2.split(im)
    if hsv:
        eq_R = cv2.equalizeHist(R)
        eq_G = cv2.equalizeHist(G)
        eq_B = cv2.equalizeHist(B)
        eq_image = cv2.merge([eq_R, eq_G, eq_B])
    else:
        eq_image = cv2.merge([R, G, B])
    
    return np.array(eq_image/255,np.float32)

def noisy(noise_typ,image,**args):
    """
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise. **args should include var, and sigma (both are numbers between 0 and 1)
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.**args should include s_vs_p and it is the ratio of salt to pepper
        and the amount of nois with respect to the image (both are numbers between 0 and 1)
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
    """
    
    if noise_typ == Noises.NOIS_GAUSS:

        row,col,ch= image.shape
        mean = 0
        var = float(args['var'])
        sigma = var**float(args['sigma'])#0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == Noises.NOIS_SALT_AND_PEPPER:
        row,col,ch = image.shape
        s_vs_p = args['s_vs_p']#0.5
        amount = args['amount']#0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == Noises.NOIS_POISSON:
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ ==Noises.NOIS_SPECKLE:
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy