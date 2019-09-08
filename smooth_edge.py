import numpy as np
import cv2, os
from tqdm import tqdm

# edge smoothing
def smooth_edge(path, img_size=256):
    file_list = os.listdir(path)
    save_path = 'dataset/smooth_cartoon_imgs'
    
    if not os.path.exists(os.path.abspath()+'/'+save_path):
        os.mkdir(save_path)

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)

    for f in tqdm(file_list):
        file_name = os.path.basename(f)

        # deal with rgb images
        rgb_img = cv2.imread(path+'/'+f)
        rgb_img = cv2.resize(rgb_img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        pad_img = np.pad(rgb_img, ((2, 2), (2, 2), (0, 0)), mode='reflect')

        # deal with gray images
        gray_img = cv2.imread(path+'/'+f, 0)
        gray_img = cv2.resize(gray_img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

        # get the edges and dilations
        edges = cv2.Canny(gray_img, 100, 200)
        dilation = cv2.dilate(edges, kernel)

        # gaussian smoothing in dilated edge areas
        result = np.copy(rgb_img)
        idx = np.where(dilation != 0)
        for i in range(np.sum(dilation != 0)):
            result[idx[0][i], idx[1][i], 0] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
            result[idx[0][i], idx[1][i], 1] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
            result[idx[0][i], idx[1][i], 2] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))
        
        cv2.imwrite(os.path.join(save_path, file_name), result)

if __name__ == "__main__":
    smooth_edge('dataset/cartoon_imgs')
