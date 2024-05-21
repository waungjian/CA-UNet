import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2

# test_mun_list = ['01', '02', '03', '04', '08', '22', '25', '29', '32', '35', '36', '38']
test_mun_list = ['02', '25', '38']

color_list = [np.array([0, 0, 0], dtype='uint8').reshape(1, 1, 3),np.array([255, 0, 0], dtype='uint8').reshape(1, 1, 3),np.array([0, 255, 0], dtype='uint8').reshape(1, 1, 3),
              np.array([0, 0, 255], dtype='uint8').reshape(1, 1, 3),np.array([255, 255, 0], dtype='uint8').reshape(1, 1, 3),
              np.array([255, 0, 255], dtype='uint8').reshape(1, 1, 3),np.array([0, 255, 255], dtype='uint8').reshape(1, 1, 3),
              np.array([255, 204, 51], dtype='uint8').reshape(1, 1, 3),np.array([234, 234, 234], dtype='uint8').reshape(1, 1, 3)]

# color_dict = {1:(0, 0, 255), 2:(0,255,0),3:(255, 0, 0), 4:(0,0,255),5:(255, 0, 255), 6:(255,255,0),7:(51, 204, 255), 8:(234,234,324)}


for test_mun in test_mun_list:
    image_path = "./model_out/predictions/case00" + str(test_mun) + "_img.nii.gz"
    pred_path = "./model_out/predictions/case00" + str(test_mun) + "_pred.nii.gz"
    gt_path = "./model_out/predictions/case00" + str(test_mun) + "_gt.nii.gz"






    image_obj = nib.load(image_path)
    pred_obj = nib.load(pred_path)
    gt_obj = nib.load(gt_path)

    print(f'Type of the image {type(image_obj)}')

    image_data = image_obj.get_fdata()
    pred_data = pred_obj.get_fdata()
    gt_data = gt_obj.get_fdata()

    type(image_data)

    height, width, depth = image_data.shape
    print(f"The image object height: {height}, width:{width}, depth:{depth}")


    for i in range(depth):
        # cv2.imwrite("./image74.png", image_data[:, :, i])
        # cv2.imwrite("./pred74.png", pred_data[:, :, i])
        # cv2.imwrite("./gt74.png", gt_data[:, :, i])

        # Define a channel to look at
        # print(f"Plotting z Layer {i} of Image")
        # plt.subplot(1,3,1)
        # plt.imshow(image_data[:, :, i]*50, cmap='gray')
        # plt.subplot(1,3,2)
        # plt.imshow(pred_data[:, :, i]*50, cmap='gray')
        # plt.subplot(1,3,3)
        # plt.imshow(gt_data[:, :, i]*50, cmap='gray')
        # plt.show()
        # plt.axis('off')
        if np.sum(gt_data[:, :, i]) > 0:
            img = image_data[:, :, i] * 255
            img = cv2.merge((img, img, img))
            # img = cv2.flip(img, 1)
            # cv2.imwrite(
            #     "./outputf/picture/pic/" + str(test_mun) + str(
            #         i) + ".png", img)

            mask = pred_data[:, :, i]
            #mask = gt_data[:, :, i]

            mask_pre = np.zeros_like(img)
            for j in range(mask_pre.shape[0]):
                for k in range(mask_pre.shape[1]):
                    if int(mask[j][k]) != 0:
                        img[j,k,:] = color_list[int(mask[j][k])]

            img = cv2.flip(img, 1)
            cv2.imwrite("./picture/RSCT-UNet/" + str(test_mun) + str(i) + ".png", img)
            cv2.imwrite("./outputf/picture/GT/" + str(test_mun) + str(i) + ".png", img)


