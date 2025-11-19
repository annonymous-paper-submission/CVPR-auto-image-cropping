
import utils
from shapely.geometry import Polygon
import pointpats 
import math
import warnings 
import seaborn as sns
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from utils import *

from scipy.spatial import distance
from scipy.fft import fftn, ifftn
from scipy.ndimage import gaussian_filter

from scipy.stats import gaussian_kde
# from utils import *
import scipy.ndimage as ndi
import seaborn as sns
import json
import pandas as pd
import os
import time
warnings.filterwarnings('ignore')

if os.path.isdir("./data"):
     print("Data folder exists")
else:
     os.makedirs("./data")
     os.makedirs("./data/img")
     os.makedirs("./data/gt")

if os.path.isdir("./plots"):
     print("Plots folder exists")
else:
     os.makedirs("./plots")
     


def centroid(center_points):
    
    hull = cv.convexHull(center_points)

    
    new_rect = cv.minAreaRect(hull)
    (x, y), (w, h), angle = new_rect
    box = cv.boxPoints(new_rect)
    box = np.int32(box) 
    coords = box
    pgon = Polygon(coords)
    
    p  = pointpats.random.poisson(pgon, size=5000)
    poission_points = []
    points_on_hull = 0
    for point in p:
        if(cv.pointPolygonTest(hull, point, False)>=0): 
            poission_points.append([int(point[0]), int(point[1])])
            points_on_hull +=1
        if(points_on_hull >= 14):
            break
    # print("poission_points>>", poission_points)
    center_x = 0
    center_y = 0


    for i in range(len(poission_points)):

        center_x += poission_points[i][0]
        center_y += poission_points[i][1]
    
    # print(points_on_hull)
    center_x = center_x / points_on_hull
    center_y = center_y / points_on_hull

    return int(center_x), int(center_y)


def saliency_aggregation(img):

    entropy_vals = normalize_zero_one(entropy(img, disk(5)))
    fourier_vals = normalize_zero_one(fourier(img))
    fourier_vals = normalize_zero_one(cv.GaussianBlur(fourier_vals, (33,33), 0))
    phot_vals = normalize_zero_one(PHOT_mod(img))
    darg_vals = normalize_zero_one(return_darg_image_main(img))
    

    aggregate_all_channels = entropy_vals+ fourier_vals + phot_vals + darg_vals  
    filtered_image = aggregate_all_channels


    aggregate_all_channels = cv.GaussianBlur(aggregate_all_channels, (33,33), 0)
    saliency = copy.copy(aggregate_all_channels)
    agg_t = np.percentile(aggregate_all_channels.flatten(), 90)
    
    _, zero_img = cv.threshold(aggregate_all_channels, thresh=agg_t, maxval=255, type=cv.THRESH_BINARY)

    
    zero_img = np.uint8(zero_img)

    # print(np.max(zero_img))
    if(np.max(zero_img) == 0):
         return -1
    contours, _ = cv.findContours(zero_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    areas = []
    for contour in contours:
        areas.append(cv.contourArea(contour))

    top_10_pct = np.percentile(areas , 90)

    filtered_image = np.zeros_like(zero_img)
    for contour in contours:
        if cv.contourArea(contour) >= top_10_pct:
            cv.drawContours(filtered_image, [contour], -1, 255, thickness=cv.FILLED)

    x, y, w, h = return_only_the_centroid_and_height_width_of_rectangle_on_convex_hull(filtered_image, saliency)
    return x, y, w, h


def saliency_aggregation_using_pmf(img):


    entropy_vals = np.array(entropy(img, disk(5)))
    entropy_vals = entropy_vals / np.sum(entropy_vals)
    fourier_vals = normalize_zero_one(fourier(img))
    fourier_vals = cv.GaussianBlur(fourier_vals, (33,33), 0)
    fourier_vals = fourier_vals / np.sum(fourier_vals)
    phot_vals = PHOT_mod(img)
    phot_vals = phot_vals / np.sum(phot_vals)
    darg_vals = return_darg_image_main(img)
    darg_vals = darg_vals / np.sum(darg_vals)
    

    aggregate_all_channels = entropy_vals+ fourier_vals + phot_vals + darg_vals  

    aggregate_all_channels = normalize_zero_one(aggregate_all_channels)
    aggregate_all_channels = cv.GaussianBlur(aggregate_all_channels, (33,33), 0)
    
    agg_t = np.percentile(aggregate_all_channels.flatten(), 90)
    
    _, zero_img = cv.threshold(aggregate_all_channels, thresh=agg_t, maxval=255, type=cv.THRESH_BINARY)

    
    zero_img = np.uint8(zero_img)


    contours, _ = cv.findContours(zero_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    areas = []
    for contour in contours:
        areas.append(cv.contourArea(contour))

    top_10_pct = np.percentile(areas , 90)

    filtered_image = np.zeros_like(zero_img)
    for contour in contours:
        if cv.contourArea(contour) >= top_10_pct:
            cv.drawContours(filtered_image, [contour], -1, 255, thickness=cv.FILLED)

    x, y, w, h = return_only_the_centroid_and_height_width_of_rectangle_on_convex_hull(filtered_image, img)
    return x, y, w, h

def main():
 
    
    img_pth = "/home/readinggroup/Desktop/Image_proc_Noman/CVPR_2025/CVPR_selected_dataset_and_images/Natural_image_dataset/MSRA_1000_randomly_selected_images/img"
    gt_pth = "/home/readinggroup/Desktop/Image_proc_Noman/CVPR_2025/CVPR_selected_dataset_and_images/Natural_image_dataset/MSRA_1000_randomly_selected_images/gt"
    imgs, gts, base_names = read_images_from_folder(img_dir=img_pth, gt_dir=gt_pth)
    imgs_color, gts_color= read_images_from_folder_color(img_dir=img_pth, gt_dir=gt_pth)
    
    
    image_counter = 0

    

    start_time = time.time()
    
    global_height = []
    global_width = []
    global_center_x = []
    global_center_y = []
    
    for image_counter in range(len(imgs)):
            width_and_heights = []
            print("Image being processed>> ", image_counter)
            print("Image Name ", base_names[image_counter])
            
            _ = return_entropy_centroid(imgs[image_counter], change_point="single")
            if(np.max(_) != -1):
                x, y , w, h  = _
                width_and_heights.append([w,h])
                entropy_single = np.array([x,y])
            else:
                entropy_single = -1

            # Entropy avg_of_two
            _ = return_entropy_centroid(imgs[image_counter], change_point="avg_of_two")

            if(np.max(_) != -1):
                x, y , w, h  = _
                width_and_heights.append([w,h])
                entropy_avg_of_two = np.array([x,y])
            else:
                entropy_avg_of_two = -1
            



            # Entropy avg of avg
            _ = return_entropy_centroid(imgs[image_counter], change_point="avg_avg")
            if(np.max(_) != -1):
                x, y , w, h  = _
                width_and_heights.append([w,h])
                entropy_avg_avg = np.array([x,y])
            else:
                entropy_avg_avg = -1
            


            # Fourier 
            _ = return_fourier_centroid(imgs[image_counter])
            if(np.max(_)!=-1 ):
                x, y, w, h = _
                width_and_heights.append([w,h])
                fourier_centroid = np.array([x,y])
            else:
                fourier_centroid = -1
            

            # Phot 
            _ = return_phot_centroid(imgs[image_counter])
            if(np.max(_) != -1):
                x, y, w, h = _
                width_and_heights.append([w,h])
                phot_centroid = np.array([x,y])
            else:
                phot_centroid = -1

            # Saliency aggregate
            _ = saliency_aggregation(imgs[image_counter])
            if(np.max(_) != -1):
                x, y, w, h = _
                width_and_heights.append([w,h])
                saliency_aggregated_centroid = np.array([x,y])
            else:
                 saliency_aggregated_centroid = -1

            # Darg
            _ = return_darg_centroids(imgs[image_counter])
            if(np.max(_) != -1):
                 x, y, w, h = _
                 width_and_heights.append([w,h])
                 darg_centroid = np.array([x,y])
            else:
                 darg_centroid = -1

            check_centroids_for_edge_cases = [entropy_single, entropy_avg_of_two, entropy_avg_avg, fourier_centroid, phot_centroid, saliency_aggregated_centroid, darg_centroid]
            center_points = [cp for cp in check_centroids_for_edge_cases if np.max(cp) != -1]
            center_points = np.array(center_points)



            if(len(np.unique(center_points, axis=0)) > 2):
                center_x, center_y = centroid(center_points)
                # print("Centers >> ", center_x, center_y)
                global_center_x.append(center_x)
                global_center_y.append(center_y)
            elif(len(np.unique(center_points, axis=0)) == 2):
                 uniq_arr = np.unique(center_points, axis=0)
                 center_x = (uniq_arr[0][0]+uniq_arr[1][0])//2
                 center_y = (uniq_arr[0][1]+uniq_arr[1][1])//2
                 global_center_x.append(center_x)
                 global_center_y.append(center_y)

            # Bounding box logic
            widths = []
            heights = []
            for k in range(len(width_and_heights)):
                widths.append(width_and_heights[k][0])
                heights.append(width_and_heights[k][1])

            mean_width = int(np.median(widths))
            mean_height = int(np.median(heights))

            
            global_height.append(mean_height)
            global_width.append(mean_width)

            
    
    global_height = np.array(global_height)
    global_width = np.array(global_width)
    global_height = global_height[global_height < 512]
    global_width = global_width[global_width < 512]

    h = np.percentile(global_height, 90)
    w = np.percentile(global_width, 90)

    glob_mean_width = int(np.max([h,w]))
    glob_mean_height = int(np.max([h,w]))
    
    


    image_counter = 0
    data_arr = []
    data_area_arr = []
    for image_counter in range(len(imgs)):
   
                center_x = global_center_x[image_counter]
                center_y = global_center_y[image_counter]
                mod_img= copy.copy(imgs_color[image_counter])
                
                start_x = center_x-(glob_mean_width//2)
                end_x = center_x+(glob_mean_width//2)
                start_y =  center_y-(glob_mean_height//2)
                end_y = center_y+(glob_mean_height//2)


                if(start_x <=0):
                    start_x = 0
                if(end_x > imgs[image_counter].shape[1]):
                    end_x = imgs[image_counter].shape[1]

                if(start_y <=0):
                    start_y = 0
                if(end_y > imgs[image_counter].shape[0]):
                    end_y = imgs[image_counter].shape[0]

                start_x = int(start_x)
                start_y = int(start_y)
                end_x = int(end_x)
                end_y = int(end_y)

                half_of_mean_width = glob_mean_width//2
                half_of_mean_height = glob_mean_height//2
                

                x_left = center_x - half_of_mean_width
                x_right = center_x + half_of_mean_width
                y_up = center_y - half_of_mean_height
                y_down = center_y + half_of_mean_height
                # left outside of the box
                if((center_x - half_of_mean_width) < 0):
                    #
                    x_left = 0
                    diff = np.abs((center_x - half_of_mean_width)) - x_left
                    
                    x_right = x_right + diff

                # right outside of the box
                if((center_x + half_of_mean_width) > 512):
                    x_right = 512
                    diff = np.abs((center_x + half_of_mean_width)) - x_right
                    x_left = x_left - diff
                    

                # Up outside of the box
                if((center_y - half_of_mean_height) < 0):
                    y_up = 0
                    diff = np.abs((center_y - half_of_mean_height)) - y_up
                    y_down = y_down + diff

                if((center_y + half_of_mean_height) > 512):
                    y_down = 512
                    diff = np.abs((center_y + half_of_mean_height)) - y_down
                    y_up = y_up - diff


                
                cv.rectangle(mod_img, (x_left, y_up),(x_right, y_down),(0,0,255), 10)
                

                sub_img = imgs_color[image_counter][y_up:y_down, x_left:x_right]
                sub_gt = gts[image_counter][y_up:y_down, x_left:x_right]
                

                data = {
                    "name": str(base_names[image_counter]),
                    "point1": [int(y_up), int(x_left)],
                    "point2": [int(y_up), int(x_right)],
                    "point3": [int(y_down), int(x_right)],
                    "point4": [int(y_down), int(x_left)],
                }

                data_area = {
                    "name": str(base_names[image_counter]),
                    "area": cropped_area_preservation_calculation(gts[image_counter], sub_gt, x_left, x_right, y_up, y_down)
                }

                data_area_arr.append(data_area)


                data_arr.append(data)
   
                cv.imwrite(f"./data/{base_names[image_counter]}", mod_img)
                cv.imwrite(f"./data/img/{base_names[image_counter]}", sub_img)
                cv.imwrite(f"./data/gt/{base_names[image_counter]}", sub_gt)


    area_overlap_vals = [data['area'] for data in data_area_arr]
    area_overlap_vals = np.array(area_overlap_vals)
    area_test = [0.70, 0.80, 0.90, 0.95]
   

    collective_area_statistics = []
    for i in range(len(area_test)):
        ar_stats = {
             f"Num_vals_greater_than_{area_test[i]}" : str(np.sum(area_overlap_vals>=area_test[i]))
        }
        collective_area_statistics.append(ar_stats)
    
    collective_area_statistics.append({"Reduced_dimension": f"{np.abs(y_down - y_up)}x{np.abs(x_left-x_right)}"})
    end_time = time.time()
    # Calculate the elapsed time in seconds
    elapsed_seconds = end_time - start_time
    # Convert elapsed time to minutes
    elapsed_minutes = elapsed_seconds / 60
    # Print the elapsed time in minutes
    print(f"Elapsed time: {elapsed_minutes:.2f} minutes")

    collective_area_statistics.append({"Elapsed_time" : f"{elapsed_minutes:.2f} minutes"})

    global_height = global_height.tolist()
    global_width = global_width.tolist()
    data_height_width = [{
         "Heights" : global_height,
         "Widths" : global_width
    }]
    with open("output.json", "w") as json_file:
            json.dump(data_arr, json_file, indent=4)


    with open(f"area_statistics_{len(imgs)}.json", "w") as json_file:
            json.dump(data_area_arr, json_file, indent=4)

    with open(f"collective_area_statistics_{len(imgs)}.json", "w") as json_file:
            json.dump(collective_area_statistics, json_file, indent=4)
    with open(f"Length_distribution_{len(imgs)}.json", "w") as json_file:
            json.dump(data_height_width, json_file, indent=4)
                    
    data_height = pd.Series(global_height)
    data_width = pd.Series(global_width)



    

    # plt.plot(heightx, heighty)
    data_height.plot.density(color='red', label ="height", lw=2)
    data_width.plot.density(color='blue', label='width', lw = 2)
    # plt.title('Density Plot of Sample Data')
    plt.axvline(x=glob_mean_width, color='green', lw=2)
    plt.xlabel('ARM length')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f"./plots/arm_length_distribution_{len(imgs)}_num_of_images.svg", format='svg', bbox_inches='tight', pad_inches=0)
    plt.figure()
    # plt.hist(global_height, bins=8, histtype='step', color='blue')
    global_height = np.array(global_height)
    global_width = np.array(global_width)
    sns.histplot(data=global_height , edgecolor='black', lw = 2)
    plt.ylabel('Counts')
    plt.xlabel('Height Bins')
    plt.savefig(f"./plots/arm_height_histogram_{len(imgs)}_num_of_images.svg", format='svg', bbox_inches='tight', pad_inches=0)
    plt.figure()
    
    sns.histplot(data=global_width, edgecolor='black', lw = 2)
    plt.ylabel('Counts')
    plt.xlabel('Width Bins')
    plt.savefig(f"./plots/arm_width_histogram_{len(imgs)}_num_of_images.svg", format='svg', bbox_inches='tight', pad_inches=0)
    
    plt.show()


if __name__ == "__main__":
    main()