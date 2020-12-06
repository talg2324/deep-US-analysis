import numpy as np
import pydicom
import os

def read_ims(data_path):
    counter = 1
    for i in range(len(os.listdir(data_path))//2):
        vid_path = data_path + 'D' + f'{counter:02}' + '.dcm'
        label_vid_path = data_path + 'D' + f'{counter:02}' + '_ground_truth.dcm'

        ims_dcm = pydicom.dcmread(vid_path)
        labels_dcm = pydicom.dcmread(label_vid_path)

        ims = ims_dcm.pixel_array
        labels = labels_dcm.pixel_array
        augment_data(ims, labels)

def augment_data_loop(ims, labels):
    for i in range(ims.shape[0]):
        im = ims[i]
        label = labels[i]
        augment_image(im, label)
    return

def augment_image(im, label):
    #DOSTUFF
    return



if __name__ == "__main__":
    read_ims('./data/')