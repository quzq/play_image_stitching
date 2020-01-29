import cv2
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8), dpi=600)
img31 = cv2.imread('./images/1-1.jpg')
gray1 = cv2.cvtColor(img31,cv2.COLOR_BGR2GRAY) 
img32 = cv2.imread('./images/2-1.jpg')
gray2 = cv2.cvtColor(img32,cv2.COLOR_BGR2GRAY) 

akaze = cv2.AKAZE_create() # AKAZE検出器の生成
kp1, des1 = akaze.detectAndCompute(gray1,None) # 特徴量の検出と特徴量ベクトルの計算
kp2, des2 = akaze.detectAndCompute(gray2,None) 

bf = cv2.BFMatcher()    # BFMatcherオブジェクトの生成

matches = bf.knnMatch(des1, des2, k=2)  # Match descriptorsを生成
# matches = bf.match(des1, des2)
ratio = 0.5
good = []
for m, n in matches:
    if m.distance < ratio * n.distance:
        good.append([m])

# 対応する特徴点同士を描画
img3 = cv2.drawMatchesKnn(gray1, kp1, gray2, kp2, good[:50], None, flags=2)

plt.imshow(img3)