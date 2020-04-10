import cv2
import numpy as np
import scipy.ndimage

WHITE = 0
BLACK = 255

def showPic(pic, title):
    if pic.dtype == 'bool':
        b = np.zeros((pic.shape[0], pic.shape[1],3)) # change bool pic to RGB
        b[:, :, 0] = b[:, :, 1] = b[:, :, 2] = pic[:, :]
        b = np.where(b, WHITE, BLACK)
        pic = b.astype('uint8')
    cv2.namedWindow(title)
    while True:
        cv2.imshow(title, pic)
        key = cv2.waitKey(20)
        if key == 32:  # exit on space
            break


def get_img_from_camera():
    cv2.namedWindow("camera")
    vc = cv2.VideoCapture(0)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv2.imshow("camera", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 32:  # exit on space
            break

    # cv2.destroyWindow("camera")
    vc.release()
    return frame;


def mean_filter(pic):
    w = np.full((5, 5), 1.0 / 25)
    pic_after_conv = np.zeros(pic.shape, dtype='uint8')

    pic_after_conv[:, :, 0] = scipy.ndimage.filters.convolve(pic[:, :, 0], w)
    pic_after_conv[:, :, 1] = scipy.ndimage.filters.convolve(pic[:, :, 1], w)
    pic_after_conv[:, :, 2] = scipy.ndimage.filters.convolve(pic[:, :, 2], w)
    return pic_after_conv


def threshold(pic):
    threshold_level = 100
    a = ~((pic[:, :, 0] < threshold_level) & (pic[:, :, 1] < threshold_level) & (pic[:, :,
                                                                                2] < threshold_level))
    return a


pic = get_img_from_camera()

pic_after_conv = mean_filter(pic)
showPic(pic_after_conv, "after blur")

pic_after_threshold = threshold(pic_after_conv)
showPic(pic_after_threshold, "binary picture")

kernel = np.ones((5,5),np.uint8)

pic_after_erosion = cv2.erode(pic_after_threshold.astype('uint8'),kernel,iterations = 1).astype('bool')
showPic(pic_after_erosion, "erosion")

pic_after_dilation = cv2.dilate(pic_after_erosion.astype('uint8'),kernel,iterations = 1).astype('bool')
showPic(pic_after_dilation, "dilation")

showPic(~pic_after_dilation, "done!")