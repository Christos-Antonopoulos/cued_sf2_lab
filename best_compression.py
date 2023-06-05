""" function that takes an image and reconstructs it within the target size (5kb) for lots of variables 
    with DWT and LBT methods, returns best image reconstruction, method, and parameters"""


#requirments 
# pip install scikit-image

import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.jpeg import jpegdec, jpegenc
from scipy import optimize
from skimage.metrics import structural_similarity as ssim
from compression import DWTCompression, LBTCompression



def calculate_ssim(imageA: np.ndarray, imageB: np.ndarray) -> float:
    """
    Calculates Structural Similarity Index (SSIM) between two images.
    
    Parameters:
        imageA: Original image
        imageB: Image to compare
        
    Returns:
        ssim_index: SSIM index between two images
    """
    # The images should be in the same data range
    assert imageA.dtype == imageB.dtype, "Input images must have the same dtype."
    assert imageA.min() >= imageB.min() and imageA.max() <= imageB.max(), \
        "Input images must have the same data range."

    return ssim(imageA, imageB, data_range=imageB.max() - imageB.min())

def error(qstep: int) -> int:

    Z, h = jpegenc(image, qstep, opthuff=True, dcbits=16, log=False)
    size = Z[:, 1].sum()
    return np.sum((size - size_lim)**2)





def best_image(input_image, size_lim, DWTstages_list, DWTblocksize_list, DWTRiseRatio_list): 
    """calculates DWT and LBT for a number of different parameters"""
    #first store raw image for comparison later
    image = input_image 
    # error = np.std(Z - image)
    # opt_step = optimize.minimize_scalar(error, method="bounded", bounds=(4, 128)).x
    # vlc, hufftab = jpegenc(image, opt_step, opthuff=True, dcbits=10, log=False)

    #jpeg method SLOW
    # Z_jpeg = jpegdec(vlc, opt_step, hufftab=hufftab, dcbits=10, log=False)

    #DWT methods 
    ### nested for loops iterating through stages list, block size list and rise ratio list, each time record image parameters 
    # initialise list to append each combination for comparison. list of lists of (title, reconstructed image, size, step size, error, SSIM, DWTstages, DWTblocksize, DWTriseratio)
    DWT_combinations_list = []

    for DWTstages in DWTstages_list:
        for DWTblocksize in DWTblocksize_list:
            for DWTriseratio in DWTRiseRatio_list: 
            
                DWT = DWTCompression(DWTstages)
                M = DWTblocksize
                Y1 = DWT.compress(image)
                #might need if statement for root2 = false or true and rise ratio 
                if DWTriseratio == 'root2': # if no rise ratio parameter, marked as root2  
                    (vlc, hufftab), qs = DWT.opt_encode(Y1, size_lim=size_lim, M=M, root2=False)
                    Y = DWT.decode(vlc, qstep=qs, hufftab=hufftab, M=M, root2=False)
                else: 
                    (vlc, hufftab), qs = DWT.opt_encode(Y1, size_lim=size_lim, M=M, rise_ratio = DWTriseratio)
                    Y = DWT.decode(vlc, qstep=qs, hufftab=hufftab, M=M, rise_ratio = DWTriseratio)

                Z_DWT = DWT.decompress(Y)

                #plot_image(Z, ax=ax)
                size = vlc[:, 1].sum()
                step_size = qs # check bounds? 
                error = np.std(Z_DWT - image)
                Z = Z_DWT.astype(image.dtype) #, the astype function is used to change the data type of Z to match the data type of image
                ssim_index = calculate_ssim( image , Z)
            
                title = f"DWT method n" #can edit if needed
                useful_parameters = (title, Z_DWT, size, step_size, error, ssim_index, DWTstages, DWTblocksize, DWTriseratio)
                DWT_combinations_list.append(useful_parameters) # this may need extra tuple brackets 
    #now DWT_combinations_list should vals for each combination - find largest SSIM and return image parameters 


    return DWT_combinations_list


#load images 
lighthouse, _ = load_mat_img(img='lighthouse.mat', img_info='X')
bridge, _ = load_mat_img(img='bridge.mat', img_info='X')
flamingo, _ = load_mat_img(img='flamingo.mat', img_info='X')
comp23, _ = load_mat_img(img = 'SF2_competition_image_2023.mat', img_info='X')

lighthouse = lighthouse - 128.0
bridge = bridge - 128.0
flamingo = flamingo - 128.0
comp23 = comp23 - 128.0 

image = comp23 # flamingo breaks variation of LBT and DWT
size_lim = 40906 - 1440 - 5


#make lists for iteration 
DWTstages_list = [3,4,5,6]
DWTblocksize_list = [8, 16, 32]
DWTRiseRatio_list = [0.5, 0.68, 1, 1.2,1.3,1.4,1.5,1.75,2]

DWTparamslist = best_image(input_image = image, size_lim = size_lim, DWTstages_list = DWTstages_list, DWTblocksize_list = DWTblocksize_list, DWTRiseRatio_list = DWTRiseRatio_list)
# print(DWTparamslist)
print('------------')

for i in DWTparamslist: 
    print( i[5])
# print(f'This should be all 16 SSIMs: {DWTparamslist[:][5]}')
 # expecting 4x2x2 = 16