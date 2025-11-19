import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from skimage.filters.rank import entropy 
from skimage.morphology import disk, diamond
import ruptures as rpt
import numpy as np
from skimage.util import view_as_windows
import matplotlib.pyplot as plt
import copy
from scipy.fft import fftn, ifftn
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon
import pointpats 
from scipy.stats import gaussian_kde
from scipy.spatial import distance
import scipy.ndimage as ndi


def normalize(img):
    """
    Args: A grayscale image of NxM dimension.
    Returns: Image normalized within range [0,255]
    """
    min = np.min(img)
    max = np.max(img)
    normalized_img = (img - min) / (max - min) * ( 255.0 - 0) 
    normalized_img = np.uint8(normalized_img)
    return normalized_img


def normalize_zero_one(img):
    min = np.min(img)
    max = np.max(img)
    normalized_image = (img - min) / (max - min)

    return normalized_image

# ================================== File reading ==========================================

def read_images_from_folder(img_dir = "/home/sajid/Desktop/Image-Processing/seperate_experiments/WACV_analysis/Datasets/COD10k_20_img_toy/imgs", gt_dir= "/home/sajid/Desktop/Image-Processing/seperate_experiments/WACV_analysis/Datasets/COD10k_20_img_toy/gts"):    
    """
    Args: Input image directory and input gt directory
    Returns: Images and their corresponding GT sorted and resized to (512x512). Sorting is based on the alphabetic order of their paths.
    """

    # Get list of files and sort them
    f = sorted(os.listdir(img_dir))
    ff = sorted(os.listdir(gt_dir))

    imgs_path = []
    gt_path = []

    # Create full paths for images and masks
    for i in range(len(f)):
        imgs_path.append(os.path.join(img_dir, f[i]))
        gt_path.append(os.path.join(gt_dir, ff[i]))


    imgs = []
    gts = []
    base_names = []

    for i in range(len(imgs_path)):
        imgs.append(cv.resize((cv.imread(imgs_path[i], cv.IMREAD_GRAYSCALE)), dsize=(512,512)))
        base_names.append(os.path.basename(imgs_path[i]))
        gts.append(cv.resize((cv.imread(gt_path[i], cv.IMREAD_GRAYSCALE)), dsize=(512,512)))


    return imgs, gts, base_names



def read_images_from_folder_with_file_names(img_dir = "/home/sajid/Desktop/Image-Processing/seperate_experiments/WACV_analysis/Datasets/COD10k_20_img_toy/imgs", gt_dir= "/home/sajid/Desktop/Image-Processing/seperate_experiments/WACV_analysis/Datasets/COD10k_20_img_toy/gts"):    
    """
    Args: Input image directory and input gt directory
    Returns: Images, their corresponding GT sorted and resized to (512x512) and also their file names.
             Sorting is based on the alphabetic order of their paths.
    """
    # Get list of files and sort them
    f = sorted(os.listdir(img_dir))
    ff = sorted(os.listdir(gt_dir))

    imgs_path = []
    gt_path = []

    # Create full paths for images and masks
    for i in range(len(f)):
        imgs_path.append(os.path.join(img_dir, f[i]))
        gt_path.append(os.path.join(gt_dir, ff[i]))


    imgs = []
    gts = []

    for i in range(len(imgs_path)):
        imgs.append(cv.resize((cv.imread(imgs_path[i], cv.IMREAD_GRAYSCALE)), dsize=(512,512)))
        gts.append(cv.resize((cv.imread(gt_path[i], cv.IMREAD_GRAYSCALE)), dsize=(512,512)))


    return imgs, gts, f



def read_images_from_folder_color(img_dir = "/home/sajid/Desktop/Image-Processing/seperate_experiments/WACV_analysis/Datasets/COD10k_20_img_toy/imgs", gt_dir= "/home/sajid/Desktop/Image-Processing/seperate_experiments/WACV_analysis/Datasets/COD10k_20_img_toy/gts"):    
    """
    Args: Input image directory and input gt directory
    Returns: Color images and their corresponding GT sorted and resized to (512x512). Sorting is based on the alphabetic order of their paths.
    """

    # Get list of files and sort them
    f = sorted(os.listdir(img_dir))
    ff = sorted(os.listdir(gt_dir))

    imgs_path = []
    gt_path = []

    # Create full paths for images and masks
    for i in range(len(f)):
        imgs_path.append(os.path.join(img_dir, f[i]))
        gt_path.append(os.path.join(gt_dir, ff[i]))


    imgs = []
    gts = []

    for i in range(len(imgs_path)):
        imgs.append(cv.resize((cv.imread(imgs_path[i])), dsize=(512,512)))
        gts.append(cv.resize((cv.imread(gt_path[i])), dsize=(512,512)))


    return imgs, gts

def read_images_from_folder_no_resize(img_dir = "/home/sajid/Desktop/Image-Processing/seperate_experiments/WACV_analysis/Datasets/COD10k_20_img_toy/imgs", gt_dir= "/home/sajid/Desktop/Image-Processing/seperate_experiments/WACV_analysis/Datasets/COD10k_20_img_toy/gts"):
    """
    Args: Input image directory and input gt directory
    Returns: Images and their corresponding GT sorted. Sorting is based on the alphabetic order of their paths.
    """
    # Get list of files and sort them
    f = sorted(os.listdir(img_dir))
    ff = sorted(os.listdir(gt_dir))

    imgs_path = []
    gt_path = []

    # Create full paths for images and masks
    for i in range(len(f)):
        imgs_path.append(os.path.join(img_dir, f[i]))
        gt_path.append(os.path.join(gt_dir, ff[i]))


    imgs = []
    gts = []

    for i in range(len(imgs_path)):
        imgs.append(cv.imread(imgs_path[i], cv.IMREAD_GRAYSCALE))
        gts.append(cv.imread(gt_path[i], cv.IMREAD_GRAYSCALE))


    return imgs, gts


# ==================================Darg==========================================

def gaussian_derivative_x_and_y(img):
    x_derivative =ndi.gaussian_filter(img,sigma=2,order=[0,1],output=np.float64, mode='nearest')
    y_derivative =ndi.gaussian_filter(img,sigma=2,order=[1,0],output=np.float64, mode='nearest')

    return x_derivative, y_derivative

def gaussian_derivative_only_y(img):
    # y_derivative = ndi.gaussian_filter(img, sigma = 30, order=[1,0], output=np.float64 , mode='nearest')
    smoothed = ndi.gaussian_filter(img, sigma=30, order =[0,0])
    
    y_derivative =ndi.gaussian_filter(smoothed,sigma=2,order=[1,0],output=np.float64, mode='nearest')
    return y_derivative

def yarg_compute(img):

    x_derivative_of_img, y_derivative_of_img = gaussian_derivative_x_and_y(img)
    

    # Computing arctan. Note the input must have y derivative first
    
    arc_tan = np.arctan2(y_derivative_of_img, x_derivative_of_img)

    # Second derivative on the arc tan wrt y
    # _, yarg = gaussian_derivative_x_and_y(arc_tan)
    yarg = gaussian_derivative_only_y(arc_tan)
    

    return yarg


def darg_compute(img):
    
    row, col = img.shape
    center = (row//2, col//2)
    yarg_img_0_deg = yarg_compute(img)
    
    

    _ = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
    
    
    yarg_img_90_deg = yarg_compute(_)
    yarg_img_90_deg = cv.rotate(yarg_img_90_deg, cv.ROTATE_90_CLOCKWISE)

    
    _ = cv.rotate(img, cv.ROTATE_180) 
    
    
    yarg_img_180_deg = yarg_compute(_)
    yarg_img_180_deg = cv.rotate(yarg_img_180_deg, cv.ROTATE_180)


    _ = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    
    
    yarg_img_270_deg = yarg_compute(_)
    yarg_img_270_deg = cv.rotate(yarg_img_270_deg, cv.ROTATE_90_COUNTERCLOCKWISE)

    

    Darg = yarg_img_0_deg + yarg_img_90_deg + yarg_img_180_deg + yarg_img_270_deg
    # print(np.dtype(img[0][0]))
    return Darg


def return_darg_image_main(img):


    darg = darg_compute(img)
            
    darg_squared = darg **2


    return darg_squared



def return_darg_centroids(img):

    
    img = np.float64(img)
    darg = darg_compute(img) 
    darg_squared = darg **2
    
    darg_squared = normalize(darg_squared)
    darg_squared = np.uint8(darg_squared)
    saliency = copy.copy(darg_squared)
    # # Apply a 70% threshold on the darg squared image
    thresh = 0.65 * 255
    _,bin_img = cv.threshold(darg_squared, thresh, 255, cv.THRESH_BINARY)


    if(np.max(bin_img) == 0):
        return -1
    x, y, w, h = return_only_the_centroid_and_height_width_of_rectangle_on_convex_hull(bin_img, saliency)

    return x, y, w, h



#=====================================Entropy ===========================================

def return_entropy_img(img, change_point):
        
        """
        Args: A grayscale image of NxM dimension and a choosen change points (i.e: single change point, avg of two etc.) 
        Returns: A binary image {0,255}, through computing Entropy and contour analysis
        """
        # assert change_point in ["avg_two", "single", ]
        bin_img = thresholded_image_based_on_peak_of_entropy_from_image(img)

        # Fixed morphological kernel
        kernel = diamond(3)
        bin_img = cv.morphologyEx(bin_img, 
                                cv.MORPH_OPEN,
                                kernel,
                                iterations=2)
        
        bin_img = cv.erode(bin_img, kernel, iterations=1)


        # bin_img = cv.dilate(bin_img, kernel, iterations=2)
        contours, _ = cv.findContours(bin_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        
        areas = []
        for contour in contours:
            areas.append(cv.contourArea(contour))

        
        if(change_point == "single"):
            if(len(areas) == 0):
                return -1
            
            if(len(np.unique(areas)) == 1):
                min_area = areas[0]

            elif(len(areas) != 1):
                
                min_area = return_change_point_after_peak(areas)
            
            else:
                min_area = areas[0]

        elif(change_point == "avg_of_two"):
            if(len(areas) == 0):
                return -1
            
            if(len(np.unique(areas)) == 1):
                min_area = areas[0]

            elif(len(areas) != 1):
                
                min_area = return_avg_of_two_change_points_after_peak(areas)    

            else:
                min_area = areas[0]


        elif(change_point == "avg_avg"):

            if(len(areas) == 0):
                return -1
            
            if(len(np.unique(areas)) == 1):
                min_area = areas[0]

            elif(len(areas) != 1):
                
                min_area = return_avg_of_single_and_avg_of_two_change_points(areas)
            
            else:
                min_area = areas[0]

        # Create an empty image for the filtered result
        filtered_image = np.zeros_like(bin_img)

        # Loop through each contour and draw it if the area is above the threshold
        new_contour = []
        for contour in contours:
            if cv.contourArea(contour) >= min_area:
                cv.drawContours(filtered_image, [contour], -1, 255, thickness=cv.FILLED)
                
    
        return filtered_image



def thresholded_image_based_on_peak_of_entropy_from_image(img):
    """
        Args: A grayscale image of NxM dimension.
        Returns: A binary image {0,255}, through using the peak Entropy value from entropy histogram as threshold.
        """
    _ = entropy(img, disk(5))
    _ = normalize(_)

    hist_values, bin_edges = np.histogram(_, bins=255)
    
    max_index = np.argmax(hist_values)
    if(max_index==0):
            max_index = np.argmax(hist_values[1:])
    

    _, bin_img = cv.threshold(_, max_index, 255, cv.THRESH_BINARY)
    bin_img = np.uint8(bin_img)
    return bin_img


def return_entropy_centroid(img, change_point):


    if(change_point == "single"):
        entropy_img = return_entropy_img(img, "single")
        
        
        
    elif(change_point == "avg_of_two"):

        entropy_img = return_entropy_img(img, "avg_of_two")
    elif(change_point == "avg_avg"):
        entropy_img = return_entropy_img(img, "avg_avg")
    else:
        print("Wrong change point!!")
        return -1
    

    # Error handling, if one image is completely black then -1 will be returned
    if(np.max(entropy_img) == -1):
        return -1
    

    saliency = entropy(img, disk(5))
    saliency = normalize(saliency)
    
    x, y, w, h = return_only_the_centroid_and_height_width_of_rectangle_on_convex_hull(entropy_img, saliency)
    

    return x, y, w, h



#========================================= FOURIER ===========================================

def fourier(img):
    """
    Args: A grayscale image of NxM dimension.
    Returns: A binary image {0,255}, through binary thresholding. The threshold value can be mean or median.
    """
      
    dft = np.fft.fft2(img)
    dft = np.fft.fftshift(dft)

    magnitude_original_img1 = np.abs(dft)
    phase_img1 = np.angle(dft)
    

    magnitude_original_img1 = 20*np.log(magnitude_original_img1)

    combined0 = np.multiply(magnitude_original_img1, np.exp(1j* phase_img1))
    combined0 = np.fft.ifftshift(combined0)
    reconstructed_image = np.fft.ifft2(combined0)
    reconstructed_image = np.abs(reconstructed_image)

    reconstructed_image = normalize(reconstructed_image)

    reconstructed_image = np.uint8(reconstructed_image)
    return reconstructed_image 


def return_fourier_centroid(img):
    
    filtered_image = fourier(img)
    zero_img = cv.GaussianBlur(filtered_image, (33,33), 0)
    saliency = copy.copy(zero_img)
    pct_90 = np.percentile(zero_img.flatten(), 90)
    _,zero_img = cv.threshold(zero_img, thresh=pct_90, maxval=255, type=cv.THRESH_BINARY)
    
    if(np.max(zero_img) == 0):
        return -1

    bin_img = zero_img

    # print(np.max(zero_img))

    contours, _ = cv.findContours(bin_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    
    areas = []
    for contour in contours:
        areas.append(cv.contourArea(contour))

    # print(areas)
    top_10_pct = np.percentile(areas , 90)
    

    # Create an empty image for the filtered result
    filtered_image = np.zeros_like(bin_img)

    
    for contour in contours:
        if cv.contourArea(contour) >= top_10_pct:
            cv.drawContours(filtered_image, [contour], -1, 255, thickness=cv.FILLED)

    x, y, w, h = return_only_the_centroid_and_height_width_of_rectangle_on_convex_hull(filtered_image, saliency)
    return x, y, w, h

#===================================PHOT===========================================

def PHOT_mod(img):
    # PHOT_mod only used to return saliency images
    DFT = fftn(img)
    MAG = np.abs(DFT) 
    PHASE = DFT / MAG  

    R1 = np.abs(ifftn(MAG))  
    R2 = np.abs(ifftn(PHASE))
    Y = gaussian_filter(R2, sigma=3)  

    

    MEAN = np.mean(Y)
    VARIENCE = np.var(Y)

    inverse_covariance = np.array([[1 / VARIENCE]])

    mahalanobis_distances = [distance.mahalanobis(np.array([x]), np.array([MEAN]), inverse_covariance) for x in Y.flatten()]

    D = mahalanobis_distances
    # Thresholding in range [0,255]
    # D = [255 if x>3 else 0 for x in D]
    D = np.reshape(D, img.shape)
    
    return D

def PHOT(img):

    DFT = fftn(img)
    MAG = np.abs(DFT) 
    PHASE = DFT / MAG  

    R1 = np.abs(ifftn(MAG))  
    R2 = np.abs(ifftn(PHASE))
    Y = gaussian_filter(R2, sigma=3)  

    # print(Y.shape)

    MEAN = np.mean(Y)
    VARIENCE = np.var(Y)

    inverse_covariance = np.array([[1 / VARIENCE]])

    mahalanobis_distances = [distance.mahalanobis(np.array([x]), np.array([MEAN]), inverse_covariance) for x in Y.flatten()]

    D = mahalanobis_distances
    # Thresholding in range [0,255]
    D = [255 if x>3 else 0 for x in D]
    D = np.reshape(D, img.shape)
    
    return D



def return_phot_centroid(img):

    phot_img = PHOT(img)


    phot_img = np.uint8(phot_img)
    zero_img = cv.GaussianBlur(phot_img, (33,33), 0)

    saliency = copy.copy(zero_img)
    pct_90 = np.percentile(zero_img.flatten(), 90)
    _,zero_img = cv.threshold(zero_img, thresh=pct_90, maxval=255, type=cv.THRESH_BINARY)
    
    bin_img = zero_img

    
    contours, _ = cv.findContours(bin_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    areas = []
    for contour in contours:
        areas.append(cv.contourArea(contour))

    
    if(len(areas) == 0):
    
        return -1
    
    top_10_pct = np.percentile(areas , 90)

    # Create an empty image for the filtered result
    filtered_image = np.zeros_like(bin_img)
    for contour in contours:
        if cv.contourArea(contour) >= top_10_pct:
            cv.drawContours(filtered_image, [contour], -1, 255, thickness=cv.FILLED)



    x, y, w, h = return_only_the_centroid_and_height_width_of_rectangle_on_convex_hull(filtered_image, saliency)
    

    return x, y, w, h    

# ======================== Change point analysis =================================

def return_change_point_after_peak(arr):
    """
    Args: A 1D array of values for example contour areas.
    Returns: Returns a single arr value after the peak where the change in signal happend.
    """
    kde = gaussian_kde(arr)
    x_vals = np.linspace(np.min(arr)- 1, np.max(arr)+1, 256)
    y_vals = kde(x_vals)

    max_index = np.argmax(y_vals)
    if(max_index==0):
            max_index = np.argmax(y_vals[1:])
    
    x_vals_modified = x_vals[max_index+1:]
    y_vals_modified = y_vals[max_index+1:]


    signal = y_vals_modified.reshape(-1,1)
    algo = rpt.Dynp(model="rbf").fit(signal)
    result = algo.predict(n_bkps=1)
    
    
    return x_vals_modified[result[0]]



def return_avg_of_two_change_points_after_peak(arr):

    """
    Args: A 1D array of values for example contour areas.
    Returns: Returns a single arr value which is average of two change points after peak.
    """
    kde = gaussian_kde(arr)
    x_vals = np.linspace(np.min(arr)- 1, np.max(arr)+1, 256)
    y_vals = kde(x_vals)

    max_index = np.argmax(y_vals)
    if(max_index==0):
            max_index = np.argmax(y_vals[1:])
    
    x_vals_modified = x_vals[max_index+1:]
    y_vals_modified = y_vals[max_index+1:]


    signal = y_vals_modified.reshape(-1,1)
    algo = rpt.Dynp(model="rbf").fit(signal)
    result = algo.predict(n_bkps=2)
    avg = (x_vals_modified[result[0]] + x_vals_modified[result[1]])/2
    
    return avg


def return_avg_of_single_and_avg_of_two_change_points(arr):
    """
    Args: A 1D array of values for example contour areas.
    Returns: Returns a single arr value which is average singel and average of two change points after peak.
    """
    kde = gaussian_kde(arr)
    x_vals = np.linspace(np.min(arr)- 1, np.max(arr)+1, 256)
    y_vals = kde(x_vals)

    max_index = np.argmax(y_vals)
    if(max_index==0):
            max_index = np.argmax(y_vals[1:])
    
    x_vals_modified = x_vals[max_index+1:]
    y_vals_modified = y_vals[max_index+1:]


    signal = y_vals_modified.reshape(-1,1)
    algo = rpt.Dynp(model="rbf").fit(signal)
    result = algo.predict(n_bkps=2)
    avg_of_two_change_point = (x_vals_modified[result[0]] + x_vals_modified[result[1]])/2


    result = algo.predict(n_bkps=1)
    single_change_point = x_vals_modified[result[0]]


    return ((avg_of_two_change_point+ single_change_point)/2)




#===================================Contour Analysis ==================================
def contour_analysis(bin_img):
    """
    Args: A binary {0,255} NxM image.
    Returns: filtered image after contour analysis
    """
    contours, _ = cv.findContours(bin_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    areas = []
    for contour in contours:
        areas.append(cv.contourArea(contour))


    # min_area = return_avg_of_two_change_points_after_peak(areas)
    min_area = return_change_point_after_peak(areas)

    
    filtered_image = np.zeros_like(bin_img)

    
    for contour in contours:
        if cv.contourArea(contour) >= min_area:
            cv.drawContours(filtered_image, [contour], -1, 255, thickness=cv.FILLED)


    return filtered_image

# ======================================= Others ===========================================
def return_only_the_centroid_and_height_width_of_rectangle_on_convex_hull(bin_img, original):
    """
    Args: A binary image and its original grayscale image of NxM dimension.
    Returns: Returns the centroid calculated using intensity methods and the height and width of rectangle formed from convex hull of the binary image
    """
    rows, cols = np.where(bin_img > 0) # Numpy uses (row,col)
    points = np.column_stack((cols, rows)) # OpenCV uses (col, row)
    
    # ##print*("The number of points is>>", len(points))
    if(len(points) == 0):
         return -1
    convex_hull = cv.convexHull(points)
    
    mask_original_using_bin_img = np.zeros_like(bin_img, dtype=np.uint8)

    mask_original_using_bin_img[bin_img>0] = original[bin_img > 0]

    # ##print*(mask_original_using_entropy.shape)

    x_numerator = 0.0
    x_denominator = 0.0
    y_numerator = 0.0
    y_denominator = 0.0


    for k in range(original.shape[0]):
        for l in range(original.shape[1]):
            # k = height = rows
            # l = width = col
            y_numerator = y_numerator+ ((k)*float(mask_original_using_bin_img[k,l]))
            y_denominator = y_denominator+ mask_original_using_bin_img[k,l]


            x_numerator = x_numerator  + ((l)*float(mask_original_using_bin_img[k,l]))
            x_denominator = x_denominator + mask_original_using_bin_img[k,l]
    


    cx = int(x_numerator/ x_denominator) # Center col
    cy = int(y_numerator/ y_denominator) # Center row



    # x,y,w,h = cv.boundingRect(convex_hull)
    rect = cv.minAreaRect(convex_hull)
    (x, y), (w, h), angle = rect
    
    
    return cx,cy, w, h



def centroid_drawing(img, center_x, center_y, color):
    """
    Args: Grayscale Image, centroid locations, color of the centroid
    Returns: A grayscale iamge with a centroid drawn at a specified location, of a specified color.
    """
    cross_size = int(0.05 * img.shape[0])
    thickness = 10
    cv.line(img, (center_x - cross_size, center_y), (center_x + cross_size, center_y), color, thickness)
    cv.line(img, (center_x, center_y - cross_size), (center_x, center_y + cross_size), color, thickness)

    return img




def cropped_area_preservation_calculation(original_gt, cropped_gt, start_x, end_x, start_y, end_y):
    """
    Args: original gt, cropped gt, start and end indexes of the original image from where the cropped gt was cutoff
    Returns: Returns a ratio of how much area of the original GT was preserved in the cropped image.
    """
    # Create a black canvas same size as original
    cropped_in_orig_space = np.zeros_like(original_gt)

    # Paste cropped mask back into original space
    cropped_in_orig_space[start_y:end_y, start_x:end_x] = cropped_gt

    # Areas
    orig_area = np.count_nonzero(original_gt)
    preserved_area = np.count_nonzero(cropped_in_orig_space & original_gt)

    

    return (preserved_area/orig_area)





def filled_retained_area_from_bounding_rectangle_and_gt(rectangel_bin_mask, gt):
    """
    Args: Binary image which has a filled rectangle and original gt, both of NxM dimension
    Returns: Returns a ratio of how much area of the original GT was preserved in the cropped image.
    """
    total_gt_pixels = np.sum(gt> 0)
    retained_gt = cv.bitwise_and(rectangel_bin_mask, gt)
    retained_gt_pixels = np.sum(retained_gt > 0)

    return (retained_gt_pixels / total_gt_pixels)


def return_gt_region_from_aggregate_image(processed_image, gt):

    zeros_img = np.zeros_like(processed_image)
    zeros_img[gt > 0] = processed_image[gt>0]
    return zeros_img



def pdf(img):

    arr = img.flatten()
    kde = gaussian_kde(arr)
    x_vals = np.linspace(0, 1, 256) # Grayscale intensity zero to one
    y_vals = kde(x_vals) # Density at each of the intensity values

    return x_vals, y_vals


def return_gt_centroid(original_gt):

    gt_contours, _ =cv.findContours(original_gt, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    areas = []
    for contour in gt_contours:
        areas.append(cv.contourArea(contour))

    for contour in gt_contours:
        if cv.contourArea(contour) == np.max(areas):
            M = cv.moments(contour)
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])


    return x,y


def distance_between_centroids(original_gt_centroid, method_centroid):
    return distance.euclidean(original_gt_centroid, method_centroid)

