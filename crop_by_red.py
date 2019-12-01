import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

def resize(img, height):
    img_h, img_w = img.shape[:2]
    img = cv2.resize(img, None, fx=height / img_h, fy=height / img_h)
    return img, height / img_h

def matching(img):
	orb = cv2.ORB_create()
	kp1, des1 = orb.detectAndCompute(template, None)
	kp2, des2 = orb.detectAndCompute(img, None)
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1, des2)
	return kp1, kp2, matches

def drawMatches(img_path, match_number=10):
	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	kp1, kp2, matches = matching(img)
	print(len(matches))
	matching_result = cv2.drawMatches(template, kp1, img, kp2, matches[:match_number], None, flags=2)
	cv2.imshow("Matching result", matching_result)
	cv2.waitKey()
	cv2.destroyAllWindows()

def gamma_correct(img, gamma=2.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)

def expand_bound(a, time=5):
    w = (a[2]-a[0]) // time
    h = (a[3]-a[1]) // time
    return (a[0]-w, a[1]-h, a[2]+w, a[3]+h)

def is_overlapping(a, b):
    a = expand_bound(a)
    b = expand_bound(b)
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return x*y


def auto_merge(rects):
	for i in range(len(rects)-1):
		for j in range(len(rects)-i-1):
			if is_overlapping(rects[i], rects[i+j+1]):
				a1, b1, c1, d1 = rects[i]
				a2, b2, c2, d2 = rects[i+j+1]
				rects.remove(rects[i+j+1])
				rects.remove(rects[i])
				rects.append((min(a1,a2), min(b1,b2), max(c1,c2), max(d1,d2)))
				return auto_merge(rects)
	return rects

def get_red_blob_bounding_box(img):
    img = add_cooling_filter(img)
    tmp = gamma_correct(img.copy())

    tmp = tmp[..., 2] - 0.5 * (tmp[..., 0] + tmp[..., 1])
    tmp -= np.min(tmp)
    tmp = tmp / np.max(tmp) * 255
    tmp = tmp.astype('uint8')

    pixel_values = np.sort(tmp.ravel())
    threshold = pixel_values[int(0.999 * len(pixel_values))]
    tmp = tmp * (tmp > threshold)

    tmp[-int(0.05*tmp.shape[0]):,:] = 0
    tmp[:int(0.05*tmp.shape[0]),:] = 0
    tmp[:,:int(0.05*tmp.shape[1])] = 0
    tmp[:,-int(0.05*tmp.shape[1]):] = 0

    contours, _ = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    print(len(contours))
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]

    bounds = []
    for blob in contours:
        peri = cv2.arcLength(blob, True)
        poly = cv2.approxPolyDP(blob, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(poly)
        bounds.append((x-5, y-5, x+w+5, y+h+5))

    bounds = auto_merge(bounds)
    print(len(bounds))

    return bounds
    for (x1, y1, x2, y2) in bounds:
        cv2.rectangle(tmp, (x1, y1), (x2, y2), (255, 255, 255), -1)
        print((x2-x1)*(y2-y1), x1, y1, x2, y2)

    cv2.namedWindow('origin', cv2.WINDOW_NORMAL)
    cv2.namedWindow('red', cv2.WINDOW_NORMAL)
    cv2.imshow('origin', img)
    cv2.imshow('red', tmp)
    cv2.resizeWindow('origin', 600,600)
    cv2.resizeWindow('red', 600,600)
    cv2.waitKey()
    cv2.destroyAllWindows()


def add_cooling_filter(img):
	from scipy.interpolate import UnivariateSpline
	def _create_LUT_8UC1(x, y):
		spl = UnivariateSpline(x, y)
		return spl(range(256))

	incr_ch_lut = _create_LUT_8UC1([0, 64, 128, 192, 256], [0, 70, 140, 210, 256])
	decr_ch_lut = _create_LUT_8UC1([0, 64, 128, 192, 256], [0, 30,  80, 120, 192])
	c_b, c_g, c_r = cv2.split(img)
	c_r = cv2.LUT(c_r, decr_ch_lut).astype(np.uint8)
	c_b = cv2.LUT(c_b, incr_ch_lut).astype(np.uint8)
	cool_img = cv2.merge((c_b, c_g, c_r))

	c_h, c_s, c_v = cv2.split(cv2.cvtColor(cool_img, cv2.COLOR_BGR2HSV))
	c_s = cv2.LUT(c_s, decr_ch_lut).astype(np.uint8)
	return cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2BGR)


def origin_vs_cooling(img):
    cv2.namedWindow('', cv2.WINDOW_NORMAL)
    cv2.imshow('', np.hstack((img, add_cooling_filter(img))))
    cv2.resizeWindow('', 1600,800)
    cv2.waitKey()
    cv2.destroyAllWindows()


def area(a):
    return (a[2]-a[0])*(a[3]-a[1])


def expect_bound(a):
	w = a[2]-a[0]
	h = a[3]-a[1]
	# return (a[0]-w*13, a[1]-h//4, a[2]+w, a[3]+h//4)

	x11 = a[0]-w*13
	x12 = a[0]-h*6

	if x11<0 and x12<0:
		return (0, a[1]-h//4, a[2]+w, a[3]+h//4)
	if x11<0:
		return (a[0]-h*6, a[1]-h//4, a[0]+h, a[3]+h//4)
	if x12<0:
		return (a[0]-w*13, a[1]-h//4, a[0]+w, a[3]+h//4)
	return (a[0]-(h*6+w*13)//2, a[1]-h//4, a[0]+(h+w)//2, a[3]+h//4)

def show_image(img):
    cv2.namedWindow('', cv2.WINDOW_NORMAL)
    cv2.imshow('', img)
    cv2.resizeWindow('', 600,130)
    cv2.waitKey()
    cv2.destroyAllWindows()

	# plt.imshow(img)
	# plt.show()

def get_largest_area(img):
	return sorted(get_red_blob_bounding_box(img), key=area, reverse=True)[0]


def crop(img):
    from task1 import get_rotate_angle
    from utils import rotate

    angle = get_rotate_angle(img.copy())
    img = rotate(img, angle)
	img1, time = resize(img.copy(), 1000)
	b = get_largest_area(img1)
	x1, y1, x2, y2 = expect_bound(b)
	x1, y1, x2, y2 = int(x1/time), int(y1/time), int(x2/time), int(y2/time)
	return img[y1:y2,x1:x2]

if __name__ == '__main__':
	from os import listdir

	DIR = 'datasets/full_image'
	img_list = [DIR+'/'+f for f in listdir(DIR) if '.jpg' in f]

	for path in img_list:
		img = cv2.imread(path)
		# img1, time = resize(img.copy(), 1000)
		# b = get_largest_area(img1)
		# x1, y1, x2, y2 = expect_bound(b)
		# # x1, y1, x2, y2 = b
		# cv2.rectangle(img, (int(x1/time), int(y1/time)), (int(x2/time), int(y2/time)), (255, 255, 255), img.shape[0]//300)
		# # cv2.rectangle(img1, (b[0], b[1]), (b[2], b[3]), (255, 255, 255), img1.shape[0]//300)
		# print(img.shape)
		show_image(crop(img))
		# origin_vs_cooling(img)
