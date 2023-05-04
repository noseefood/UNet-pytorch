import matplotlib
import matplotlib.pyplot as plt
import cv2


def plot_img_and_mask(img, mask, rate_image):
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    fr = lambda x, r: (int(x.shape[1]/r), int(x.shape[0]/r))
    # mask = cv2.resize(mask, fr(img, 1) )  # 会缩小可视化的输出
    mask[:, :, -1] = mask[:, :, -1] * 255
    img_ = cv2.addWeighted(img, rate_image, mask, 1-rate_image, 0)  # 融合两张图片，rate_image为融合率
    # img_ = cv2.resize(img_, fr(img, 3) ) # 会缩小可视化的输出
    cv2.imshow('result', img_)
    
    # 如果训练时scale为0.5,则这里需要放大：
    # print(img_.shape)
    x, y = img_.shape[0:2]
    img_resized = cv2.resize(img_, (int(x) , int(y)))
    cv2.imshow('resized result', img_resized)
    cv2.imshow('original img', img)


    d = cv2.waitKey(0)
    if d == ord('q'):
        cv2.destroyAllWindows()