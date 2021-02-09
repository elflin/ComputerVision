import cv2

img = cv2.imread('/Users/evantanuwijaya/Desktop/Icon.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("test", img)
# cv2.imshow("test-gray", gray)

cv2.waitKey(0)
cv2.destroyAllWindows()