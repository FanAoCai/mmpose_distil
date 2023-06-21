import os
import numpy as np


def heatmap():
    heatmap_file_list = os.listdir('/HOME/scz3186/run/fanao/code/mmpose/heatmap')
    for file_name in heatmap_file_list:
        print(file_name)
        npz_file = np.load('/HOME/scz3186/run/fanao/code/mmpose/heatmap/' + file_name, allow_pickle=True)
        print('success!')
        print(npz_file['heatmap'].shape)

def image_file():
    image_file_list = os.listdir('/HOME/scz3186/run/fanao/code/mmpose/image_file')
    for file_name in image_file_list:
        print(file_name)
        npz_file = np.load('/HOME/scz3186/run/fanao/code/mmpose/image_file/' + file_name, allow_pickle=True)
        print('success!')
        print(npz_file['image_file'].shape)

def heatmap_image_file():
    heatmap_image_file_list = os.listdir('/HOME/scz3186/run/fanao/code/mmpose/heatmap_image_file')
    for file_name in heatmap_image_file_list:
        print(file_name)
        npz_file = np.load('/HOME/scz3186/run/fanao/code/mmpose/heatmap_image_file/' + file_name, allow_pickle=True)
        print('success!')
        print(npz_file['image_file'].shape)

def npz_fix():
    heatmap_file_list = os.listdir('/HOME/scz3186/run/fanao/code/mmpose/heatmap')
    heatmap_list = []
    for file_name in heatmap_file_list:
        npz_file = np.load('/HOME/scz3186/run/fanao/code/mmpose/heatmap/' + file_name, allow_pickle=True)['heatmap']
        for i, every_heatmap in enumerate(npz_file):
            heatmap_list.append(every_heatmap)
    heatmap = np.array(heatmap_list)
    print('heatmap success!!!')

    image_file_file_list = os.listdir('/HOME/scz3186/run/fanao/code/mmpose/image_file')
    image_file_list = []
    for file_name in image_file_file_list:
        npz_file = np.load('/HOME/scz3186/run/fanao/code/mmpose/image_file/' + file_name, allow_pickle=True)['image_file']
        for i, every_image_file in enumerate(npz_file):
            image_file_list.append(every_image_file)
    image_file = np.array(image_file_list)
    print('image_file success!!!')

    if not os.path.exists('/HOME/scz3186/run/fanao/code/mmpose/heatmap_result'):
        os.mkdir('/HOME/scz3186/run/fanao/code/mmpose/heatmap_result')
    np.savez_compressed('/HOME/scz3186/run/fanao/code/mmpose/heatmap_result/heatmap_result.npz', heatmap = heatmap, image_file = image_file)

if __name__ == '__main__':
    # heatmap()
    # image_file()
    # heatmap_image_file()
    npz_fix()

