import os, cv2, glob
import numpy as np
import matplotlib.pyplot as plt
#import thinning

def createExtractedImages(folder_path, choseColor, output_folder):

    if not os.path.isdir(folder_path):
        print ("image folder not found!")
        return

    if not os.path.isdir(output_folder):
        print ("output folder not found!")
        return

    files = glob.glob('%s/*.png' % folder_path)

    for file_path in files:
        print("reading image: "+file_path)
        image = cv2.imread(file_path)
        new_image = extractPixelFromImage(image, np.array(choseColor))
        print("saving converted image")
        cv2.imwrite(output_folder+'/'+os.path.basename(file_path), new_image)

def extractPixelFromImage(img, choseColor):
    print("begin to convert image........")
    height, width, channels = img.shape
    new_image = np.zeros((height, width, 1), np.uint8)
    for y in range(height):
        for x in range(width):
            pixel = img[y,x]
            if np.array_equal(choseColor,pixel):
                new_image[y,x] = 255
            else:
                new_image[y,x] = 0
    print("converting image finished!")
    return new_image


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


class GleisPoints:
    def __init__(self, img):
        self._img = img
        self._img_h, self._img_w = img.shape
        self.cal_img = np.zeros((self._img_h + 2, self._img_w + 2))
        self.cvt_img = np.zeros((self._img_h, self._img_w))
        self._gleis_points = []
        self._gleis_grenze_points = []
        self._min_x = 5000
        self._max_x = 0
        self._min_y = 5000
        self._max_y = 0
        self.initGleisPointsAndCalSpace()
        self.initGrenzePoints()

    def initGleisPointsAndCalSpace(self):
        for y in range(self._img_h):
            for x in range(self._img_w):
                if image[y, x] == 255:
                    self.initMinAndMax(y,x)
                    self._gleis_points.append({'x': x, 'y': y})
                    self.cal_img[y + 1, x + 1] = 1
                    self.cvt_img[y,x] = 1

        for x in range(self._img_w+2):
            self.cal_img[0, x] = 1
            self.cal_img[self._img_h+1, x] = 1

        for y in range(self._img_h+2):
            self.cal_img[y, 0] = 1
            self.cal_img[y, self._img_w+1] = 1


    def initMinAndMax(self,y, x):
        if x < self._min_x:
            self._min_x = x
        if x > self._max_x:
            self._max_x = x
        if y < self._min_y:
            self._min_y = y
        if y > self._max_y:
            self._max_y = y

    def initGrenzePoints(self):
        for point_position in self._gleis_points:
            if self.filterPointsWithNachbarn(point_position['x'], point_position['y']):
                self._gleis_grenze_points.append(point_position)

    def filterPointsWithNachbarn(self, x, y, minNachbarnNum = 2,  maxNachbarnNum = 8):
        nachbarn = self.getNachbarnPosition(x + 1, y + 1)
        numOfPointsWithColor = 0
        for point in nachbarn:
            numOfPointsWithColor += self.cal_img[point['y'], point['x']]

        if numOfPointsWithColor >= minNachbarnNum and numOfPointsWithColor <= maxNachbarnNum :
            return True

        return False

    def getNachbarnPosition(self,x, y):
        points = []
        points.append({'x': x - 1, 'y': y - 1})
        points.append({'x': x, 'y': y - 1})
        points.append({'x': x + 1, 'y': y - 1})
        points.append({'x': x - 1, 'y': y})
        points.append({'x': x + 1, 'y': y})
        points.append({'x': x - 1, 'y': y + 1})
        points.append({'x': x, 'y': y + 1})
        points.append({'x': x + 1, 'y': y + 1})
        return points

    def getGleisPoints(self):
        return self._gleis_points

    def getGleisGrenzePoints(self):
        return self._gleis_grenze_points

    def getMinPointsX(self):
        return self._min_x

    def getMaxPointsX(self):
        return self._max_x

    def getMinPointsY(self):
        return self._min_y

    def getMaxPointsY(self):
        return self._max_y

    def getCvtImage(self):
        return self.cvt_img

    def savePointsAsImg(self, image_name, points):
        grenzImg = np.zeros((self._img_h, self._img_w, 1), np.uint8)
        for point in points:
            grenzImg[point['y'], point['x']] = 255
        cv2.imwrite(image_name, grenzImg)


    def getExprimentPoints(self, points):
        exprimentPointsX = []
        exprimentPointsY = []
        for point in points:
            exprimentPointsX.append(point['x'])
            exprimentPointsY.append(point['y'])

        return exprimentPointsX, exprimentPointsY

    def getExprimentStartXAndEndX(self, points):
        startX = 0
        endX = 5000
        for point in points:
            if point['y'] == self._min_y and point['x'] > startX:
                startX = point['x']

            if point['y'] == self._max_y and point['x'] < endX:
                endX = point['x']

        return startX -45 , endX + 45
'''
class Thinning:

    def __init__(self, imgname):
        self._img = cv2.imread(imgname)
        self._h, self._w = self._img.shape
        self._img_gray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        self._img_thin = thinning.guo_hall_thinning(self._img_gray)


    def saveResult(self, resultname):

        for y in range(self._h):
            for x in range(self._w):
                if self._img_thin[y, x] == 255:
                    self._img[y, x] = (0,0,0)

        cv2.imwrite(resultname, self._img)
'''
if __name__ == "__main__":

    image = cv2.imread('test.png')
    new_img = extractPixelFromImage(image, [255,0,255])

    kernel = np.ones((5, 5), np.uint8)
    erode = cv2.erode(new_img, kernel, iterations=1)
    closing = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, kernel)
    #gp = GleisPoints(gradient)
    #points_left_x, points_left_y = gp.getExprimentPoints(gp.getGleisGrenzePoints())
    #plt.plot(points_left_y, points_left_x, 'ro')
    #plt.imshow(gradient)
    #plt.show()
    #cv2.imwrite('gradient.png', gradient)
    edges = auto_canny(gradient)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,minLineLength, maxLineGap)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), (255,0, 0), 2)
    '''
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(image, (x1, y1), (x2, y2), (255,0, 0), 2)
    '''

    cv2.imwrite('houghlines.png', image)
