import numpy as np
import pydicom
import os
from matplotlib import pyplot as plt
import cv2

def read_ims(data_path):
    counter = 1
    for i in range(len(os.listdir(data_path+'/input/'))):
        vid_path = data_path + '/input/D' + f'{counter:02}' + '.dcm'
        label_vid_path = data_path + '/output/D' + f'{counter:02}' + '_ground_truth.dcm'

        ims_dcm = pydicom.dcmread(vid_path)
        labels_dcm = pydicom.dcmread(label_vid_path)

        ims = ims_dcm.pixel_array
        out_labels = labels_dcm.pixel_array
        mid_labels = generate_mid_labels(out_labels)

        np.save('./data/%d_in.npy'%(i), ims)
        np.save('./data/%d_out.npy'%(i), out_labels)
        np.save('./data/%d_mid.npy'%(i), mid_labels)

def playback_vid(ims):
    for i in range(ims.shape[0]):
        cv2.imshow('video', ims[i,:,:])
        cv2.waitKey(10)

def generate_mid_labels(labels):
    z,y,x = labels.nonzero()

    ROI = np.zeros_like(labels)
    ROI[z.min():z.max(), y.min():y.max(), x.min():x.max()] = 1

    return ROI


if __name__ == "__main__":
    read_ims('./data/')