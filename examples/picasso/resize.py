import cv2

src = cv2.imread('lena_512.png', cv2.IMREAD_COLOR)

dst = cv2.resize(src, dsize=(256,256), interpolation=cv2.INTER_LINEAR)

cv2.imshow('src',src)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('lena_256.png', dst)
