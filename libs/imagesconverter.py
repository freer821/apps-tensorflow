import os, cv2, glob, sys
import numpy as np
import matplotlib.pyplot as plt
import thinning

sys.setrecursionlimit(1500)

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
    new_image = np.zeros((height, width), np.uint8)
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

class PointsClassifier:
    def __init__(self, img):
        self._img = img
        self._img_h, self._img_w = img.shape
        self.cal_img = np.zeros((self._img_h + 2, self._img_w + 2))
        self.cvt_img = np.zeros((self._img_h, self._img_w))
        self._all_points = []
        self._points_groups = []
        self.initImgSpace()

    def initImgSpace(self):
        print("begin to init image space!")
        for y in range(self._img_h):
            for x in range(self._img_w):
                if self._img[y, x] == 255:
                    self.cal_img[y + 1, x + 1] = 1
                    self.cvt_img[y,x] = 1
                    self._all_points.append([x,y])


    def findNeighborsOfPoint(self, po, neighbors, pointsbygroup):
        neighborsPositions = self.getNeighborPosition(po[0] + 1, po[1] + 1)
        points_np_array = np.array(pointsbygroup)
        for point in neighborsPositions:
            orig_p = [point[0] - 1, point[1] - 1]
            if self.cal_img[point[1], point[0]] == 1 and len(np.where((points_np_array == (orig_p[0], orig_p[1])).all(axis=1))[0]) == 0:
                pointsbygroup.append(orig_p)
                neighbors.append(orig_p)

    def getNeighborPosition(self,x, y):
        points = []
        points.append([x - 1, y - 1])
        points.append([ x, y - 1])
        points.append([x + 1, y - 1])
        points.append([x - 1, y])
        points.append([x + 1, y])
        points.append([x - 1, y + 1])
        points.append([x, y + 1])
        points.append([ x + 1, y + 1])
        return points

    def getAllPoints(self):
        return self._all_points

    def addNewPointsGroup(self, points):
        self._points_groups.append(points)

    def getPointsGroups(self):
        return self._points_groups


    def savePointsAsImg(self, image_name, points):
        print("begin to save image!")
        grenzImg = np.zeros((self._img_h, self._img_w, 1), np.uint8)
        for point in points:
            grenzImg[point[1], point[0]] = 255
        cv2.imwrite(image_name, grenzImg)

    def printPoints(self, points):
        for p in points:
            self.printPoint(p)

    def printPoint(self, point):
        print("x: {0}, y: {1}".format(point[0], point[1]))

if __name__ == "__main__":

    image = cv2.imread('test.png')
    new_img = extractPixelFromImage(image, [255,0,255])

    pc = PointsClassifier(new_img)
    first_point = pc.getAllPoints()[0]
    neighbors = []
    neighbors_group = []
    neighbors.append(first_point)
    neighbors_group.append(first_point)

    print("begin to find all neighbors!")
    while len(neighbors) > 0:
        tmp_neighbors = []
        for point in neighbors:
            results = pc.findNeighborsOfPoint(point,tmp_neighbors, neighbors_group)

        neighbors = tmp_neighbors
        print ("find {0} new neighbors".format(len(neighbors)))

    pc.addNewPointsGroup(neighbors_group)
    for point in pc.getPointsByGroup():
        image[point[1], point[0]] = [255,255,0]

    print("save one railway!")
    cv2.imwrite('ouput.png', image)


