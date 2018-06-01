import numpy as np
import cv2

def resize_keep_ratio(img, size, mode=0, interpolation=cv2.INTER_LINEAR):
    r"""Resize the input PIL Image to the given size.
    Args:
        img (Array Image): Image to be resized.
        size (int): Desired output size.
        mode (int, optional): Desired mode. 
            if mode=='max', max(w, h) ->  size
            if mode=='min', min(w, h) ->  size
            if mode=='mean', mean(w, h) ->  size
            Default is 0
    Returns:
        Array Image: Resized image.
    """
    assert mode in ['max', 'min', 'mean'], \
        'Resize_keep_ratio mode should be either max, min, or mean'
        
    srcH, srcW = img.shape[0:2]
    if (srcW < srcH and mode == 'max') or (srcW > srcH and mode == 'min'):
        dstH = size
        dstW = int(float(size) * srcW / srcH)
    elif (srcW > srcH and mode == 'max') or (srcW < srcH and mode == 'min'):
        dstH = size
        dstW = int(float(size) * srcW / srcH)
    else: # mode == 'mean'
        scale = np.mean((srcH, srcW)) / size
        dstH, dstW = [srcH*scale, srcW*scale]
    
    return cv2.resize(img, (dstW, dstH), interpolation)


def pad(img, padding, value=0, borderType=cv2.BORDER_CONSTANT):
    '''
    Based on `cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value)`
    '''
    return cv2.copyMakeBorder(img, padding[0], padding[1], padding[2], padding[3], borderType, value)
    

def center_crop(img, h, w):
    width, height = img.shape[0:2]
    y1 = int(round((height - h) / 2.))
    x1 = int(round((width - w) / 2.))
    return img[y1:y1+h, x1:x1+w].copy()


# def adjust_brightness(img, brightness_factor):
#     """Adjust brightness of an Image.
#     Args:
#         img (PIL Image): PIL Image to be adjusted.
#         brightness_factor (float):  How much to adjust the brightness. Can be
#             any non negative number. 0 gives a black image, 1 gives the
#             original image while 2 increases the brightness by a factor of 2.
#     Returns:
#         PIL Image: Brightness adjusted image.
#     """
#     if not _is_pil_image(img):
#         raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

#     enhancer = ImageEnhance.Brightness(img)
#     img = enhancer.enhance(brightness_factor)
#     return img


# def adjust_contrast(img, contrast_factor):
#     """Adjust contrast of an Image.
#     Args:
#         img (PIL Image): PIL Image to be adjusted.
#         contrast_factor (float): How much to adjust the contrast. Can be any
#             non negative number. 0 gives a solid gray image, 1 gives the
#             original image while 2 increases the contrast by a factor of 2.
#     Returns:
#         PIL Image: Contrast adjusted image.
#     """
#     if not _is_pil_image(img):
#         raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

#     enhancer = ImageEnhance.Contrast(img)
#     img = enhancer.enhance(contrast_factor)
#     return img


# def adjust_saturation(img, saturation_factor):
#     """Adjust color saturation of an image.
#     Args:
#         img (PIL Image): PIL Image to be adjusted.
#         saturation_factor (float):  How much to adjust the saturation. 0 will
#             give a black and white image, 1 will give the original image while
#             2 will enhance the saturation by a factor of 2.
#     Returns:
#         PIL Image: Saturation adjusted image.
#     """
#     if not _is_pil_image(img):
#         raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

#     enhancer = ImageEnhance.Color(img)
#     img = enhancer.enhance(saturation_factor)
#     return img


# def adjust_hue(img, hue_factor):
#     """Adjust hue of an image.
#     The image hue is adjusted by converting the image to HSV and
#     cyclically shifting the intensities in the hue channel (H).
#     The image is then converted back to original image mode.
#     `hue_factor` is the amount of shift in H channel and must be in the
#     interval `[-0.5, 0.5]`.
#     See `Hue`_ for more details.
#     .. _Hue: https://en.wikipedia.org/wiki/Hue
#     Args:
#         img (PIL Image): PIL Image to be adjusted.
#         hue_factor (float):  How much to shift the hue channel. Should be in
#             [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
#             HSV space in positive and negative direction respectively.
#             0 means no shift. Therefore, both -0.5 and 0.5 will give an image
#             with complementary colors while 0 gives the original image.
#     Returns:
#         PIL Image: Hue adjusted image.
#     """
#     if not(-0.5 <= hue_factor <= 0.5):
#         raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

#     if not _is_pil_image(img):
#         raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

#     input_mode = img.mode
#     if input_mode in {'L', '1', 'I', 'F'}:
#         return img

#     h, s, v = img.convert('HSV').split()

#     np_h = np.array(h, dtype=np.uint8)
#     # uint8 addition take cares of rotation across boundaries
#     with np.errstate(over='ignore'):
#         np_h += np.uint8(hue_factor * 255)
#     h = Image.fromarray(np_h, 'L')

#     img = Image.merge('HSV', (h, s, v)).convert(input_mode)
#     return img


# def adjust_gamma(img, gamma, gain=1):
#     r"""Perform gamma correction on an image.
#     Also known as Power Law Transform. Intensities in RGB mode are adjusted
#     based on the following equation:
#     .. math::
#         I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}
#     See `Gamma Correction`_ for more details.
#     .. _Gamma Correction: https://en.wikipedia.org/wiki/Gamma_correction
#     Args:
#         img (PIL Image): PIL Image to be adjusted.
#         gamma (float): Non negative real number, same as :math:`\gamma` in the equation.
#             gamma larger than 1 make the shadows darker,
#             while gamma smaller than 1 make dark regions lighter.
#         gain (float): The constant multiplier.
#     """
#     if not _is_pil_image(img):
#         raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

#     if gamma < 0:
#         raise ValueError('Gamma should be a non-negative real number')

#     input_mode = img.mode
#     img = img.convert('RGB')

#     gamma_map = [255 * gain * pow(ele / 255., gamma) for ele in range(256)] * 3
#     img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

#     img = img.convert(input_mode)
#     return img
