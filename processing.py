import os

import cv2
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt
from functools import partial
plt.rcParams['image.cmap'] = 'gray'


def defog(img):
    from defogging import Defog
    df = Defog()
    df.read_array(img, 255)
    df.defog()
    
    return df.get_array(255)

def denoise(img):
    
    # smooth the image with alternative closing and opening
    # with an enlarging kernel
    morph = img.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # take morphological gradient
    gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)

    # split the gradient image into channels
    image_channels = np.split(np.asarray(gradient_image), 3, axis=2)

    channel_height, channel_width, _ = image_channels[0].shape

    # apply Otsu threshold to each channel
    for i in range(0, 3):
        _, image_channels[i] = cv2.threshold(~image_channels[i], 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        image_channels[i] = np.reshape(image_channels[i], newshape=(channel_height, channel_width, 1))

    # merge the channels
    image_channels = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)
    
    mask = np.min(image_channels, axis=2) < 30 
    img[mask] = 0
    
    return img

def gamma_correct(img, gamma=2.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)

def auto_gamma(img, clip=0.2):
    def auto_gamma_gray(img, clip=clip):
        hist_size = 256
        hist = cv2.calcHist(img, [0], None, [hist_size], (0, hist_size), accumulate=False)
        accumulator = np.cumsum(hist)
        clip *= (accumulator[-1] / 100.0)
        clip /= 2.0
        min_gray = next(x for x in accumulator if x > clip)
        max_gray = next(x for x in accumulator if x >= accumulator[-1] - clip) - 1
        
        input_range = max_gray - min_gray
        
        alpha = (hist_size - 1) / input_range
        beta = - min_gray * alpha
        
        img = img * alpha + beta
        
        return img
    
    if len(img.shape) <= 2 or img.shape[2] == 1:
        img = auto_gamma_gray(img)
    else:
        for i in range(3):
            img[:,:,i] = auto_gamma_gray(img[:,:,i])
    
    return img.astype('uint8')

def sharpen(img):
    # Create our shapening kernel, it must equal to one eventually
    kernel_sharpening = np.array([[-1,-1,-1], 
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
    # applying the sharpening kernel to the input image & displaying it.
    sharpened = cv2.filter2D(img, -1, kernel_sharpening)
    
    return img

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def increase_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(20,20))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def denoise_cv(img):
    # img = cv2.fastNlMeansDenoising(origin_img, h=27)
    img = cv2.fastNlMeansDenoisingColored(img, None, 13, 13, 7, 21)

    return img

def clahe(img, clipLimit=3.0, tileGridSize=(10, 17)):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img = clahe.apply(img)
    
    return img

def channel_min(img):
    img = np.min(img, axis=-1)
    
    return img

def thresh(img, block_size=5):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                cv2.THRESH_BINARY, block_size, 1)
    # plt.imshow(img)
    # plt.show()
    # for i in range(3):
    #     img_ = cv2.adaptiveThreshold(img[:,:,i], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                                 cv2.THRESH_BINARY, 301, 1)
    #     plt.imshow(img_)
    #     plt.show()
    
    return img

def global_hist_equalize(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    
    return img

bilateral_blur = cv2.bilateralFilter

if __name__ == '__main__':
    img_paths = list(paths.list_images('image_cropped'))
    # img_paths = [
    #              # r'image_cropped\25.jpg_defogged.bmp',
    #              r'image_cropped\25.jpg',
    #              r'image_cropped\26.jpg',
    #              r'image_cropped\37.jpg',
    #              ]

    operators = [
                 # [increase_contrast, 'increase contrast'],
                 # [defog, 'defog'],
                 # [denoise, 'denoise'],
                 # [white_balance, 'white balance'],
                 # [denoise_cv, 'fast n1 mean denoise'],
                 [bilateral_blur, 'bilateral blur'],
                 [sharpen, 'sharpen'],
                 # [denoise, 'denoise'],
                 # [thresh, 'thresh'],
                 # [partial(gamma_correct, gamma=2.5), 'gamma'],
                 # [partial(auto_gamma, clip=0.2), 'auto gamma'],
                 # [channel_min, 'min channel-wise'],
                 # [partial(clahe, clipLimit=3.0, tileGridSize=(17, 17)), 'clahe'],
                 # [global_hist_equalize, 'global hist equalize']
    ]

    row, col = 4, 3

    for img_path in img_paths:
    # for img_path in ['new_images/5.jpg']:
        origin_img = cv2.imread(img_path)

        img = origin_img.copy()
        plt.subplot(row, col, 1, title='original')
        plt.imshow(img)

        for i, (op, title) in enumerate(operators):
            # img = op(origin_img.copy())
            img = op(img)
            plt.subplot(row, col, i+2, title=title)
            plt.imshow(img)
            
        plt.show()









    # def process(img):
    #     origin_img = img.copy()
    #     plt.subplot('331')
    #     plt.imshow(origin_img)
    # 
    #     # img = gamma_correct(origin_img, 2.5)
    #     # img = white_balance(img)
    #     img = auto_gamma(origin_img)
    #     plt.subplot('332', title='gamma_corrected')
    #     plt.imshow(img)
    # 
    #     img = increase_contrast(img)
    #     plt.subplot('333', title='contrast')
    #     plt.imshow(img)
    # 
    #     img = defog(img)
    #     plt.subplot('334', title='defog')
    #     plt.imshow(img)
    # 
    #     # img = denoise(img)
    #     img = denoise_cv(img)
    #     plt.subplot('335', title='denoise')
    #     plt.imshow(img)
    # 
    #     # img = gamma_correct(img, 1/2.5)
    #     # plt.subplot('336', title='gamma_uncorrected')
    #     # plt.imshow(img)
    # 
    #     img = clahe(img)
    #     img = thresh(img)
    #     plt.subplot('337', title='threshed')
    #     plt.imshow(img)
    # 
    #     return img
    # 
    # 
    # for p in img_paths[:1]:
    #     img = cv2.imread(p)
    # 
    #     img = process(img)
    #     plt.show()