# File for Base class to store fingerprint


import math
from math import sqrt
import numpy as np
from skimage.morphology import skeletonize
import cv2 as cv
import fingerprint_enhancer
import os
from pathlib import Path
block_size =15
def get_line_ends(i, j, W, tang):
    index_one = (-W/2) * tang + j + W/2
    index_two = (W/2) * tang + j + W/2
    index_three =i + W/2 + W/(2 * tang)
    index_four= i + W/2 - W/(2 * tang)
    if -1 <= tang and tang <= 1:
        begin = (int(index_one), i)
        end_2=i+W
        end = (int(index_two), end_2)
    else:
        begin_1= j + W//2
        begin = (begin_1, int(index_three))
        end_1= j - W//2
        end = (end_1, int(index_four))
    return (begin, end)

def interactive_display(window_label, image):
    print("Display image function")
    cv.imshow(window_label, image)
    print("Image is loading.........................")
    while 1:
        key = cv.waitKey(0) & 0xFF
		# wait for ESC key to exit
        if key == 27:
            cv.destroyAllWindows()
            break
    cv.destroyAllWindows()

def apply_mask(image,seg_mask):
    kernel_open_close = cv.getStructuringElement(cv.MORPH_RECT,(2*block_size, 2*block_size))
    image= image.copy()
    print("Dilation")
    seg_mask = cv.morphologyEx(seg_mask, cv.MORPH_CLOSE, kernel_open_close)
    print("erosion")
    seg_mask = cv.morphologyEx(seg_mask, cv.MORPH_OPEN, kernel_open_close)
    image[seg_mask == 0] = 255
    return image

def seg_mask(image):
    (x,y) = image.shape
    #print(x,y)
    seg_mask= np.ones((x,y))
    grey_variance= np.var(image)*0.1
    for i in range(0, x, block_size):
        for j in range(0,y,block_size):
            end_i = min(x, i + block_size)
            end_j = min(y, j + block_size)
            local_variance = np.var(image[i: end_i, j: end_j])
            if local_variance <= grey_variance:
                seg_mask[i: end_i, j: end_j] = 0
    return seg_mask
def perform_normalisation(image):
    m0_mean= 127.0
    v0_variance = 5000.0
    m1_mean = np.mean(image)
    v1_variance =np.var(image)
    deviation= ((v0_variance*(image - m1_mean)**2)/v1_variance)**0.5
    if np.where(image > m1_mean):
        nor_img= m0_mean+ deviation
    else :
        nor_img= m0_mean- deviation
    return nor_img


def ridge_orientation_calculation(image,original_image):
    # Ridge orientation calculation
    x= original_image.shape[0]
    y= original_image.shape[1]
    directions_x = np.zeros((x,y))
    directions_y = np.zeros((x,y))
    grad_x = cv.Sobel(image/255, cv.CV_64F, 0, 1, ksize=3)
    grad_y = cv.Sobel(image/255, cv.CV_64F, 1, 0, ksize=3)
    div= block_size//2
    gaussian_blur_kernel_size = (2*block_size+1, 2*block_size+1)
    gaussian_std = 1.0
    for i in range(x):
        for j in range(y):
            start_i = max(0, i-div)
            start_j = max(0, j-div)
            end_i = min(i+div, x)
            end_j = min(j+div, y)
            f=grad_x[start_i: end_i, start_j: end_j]
            g= grad_y[start_i: end_i, start_j: end_j]
            directions_x[i, j] = np.sum(2*f*g)
            directions_y[i, j] = np.sum(f**2-g**2)
        
        gaussian_local_directions_x = cv.GaussianBlur(directions_x, gaussian_blur_kernel_size, gaussian_std)
        gaussian_local_directions_y = cv.GaussianBlur(directions_y, gaussian_blur_kernel_size, gaussian_std)
        orientation_map = 0.5*(np.arctan2(gaussian_local_directions_x, gaussian_local_directions_y)+np.pi)
        #print(orientation_map)
        orientation_image = cv.cvtColor((image).astype(np.uint8), cv.COLOR_GRAY2RGB)
		# self.orientation_image = np.ones(self.normalized_image.shape)
        for i in range(0, original_image.shape[0],block_size):
            for j in range(0, original_image.shape[1], block_size):
                end_i = min(original_image.shape[0], i+block_size)
                end_j = min(original_image.shape[1], j+block_size)
                line_direction = np.average(orientation_map[i:end_i, j:end_j])
                begin, end = get_line_ends(i, j,block_size, math.tan(line_direction))
                img= cv.line(orientation_image, begin, end, (255, 0, 0), 1)
    #interactive_display("",img)
    return orientation_map

def calculate_minutae(image,original_img,orientation_map):
    minutiae = {}
    minutiae_img = cv.cvtColor((255*image).astype(np.uint8), cv.COLOR_GRAY2RGB)
    x= original_img.shape[0]
    y= original_img.shape[1]
    for i in range(1,x-1):
        for j in range(1,y-1):
            index_1=i
            index_2=j
            value = cal(index_1, index_2, image)
            if value == 1 or value == 3:
                minutiae[(i, j)] = (value,orientation_map[i, j])
    return minutiae,minutiae_img
        
def cal(i, j, img):
    value= img[i, j]
    cnt=0
    if value == 0.0:
        
        offsets = [(-1, -1), (-1, 0), (-1, 1),  # p1 p2 p3
				   (0, 1),  (1, 1), (1, 0),        # p8    p4
				  	(1, -1), (0, -1), (-1, -1)] 	# p7 p6 p5
        
        pixel_values = [img[i+x, j+y] for x, y in offsets]
        cnt=cnt+1
        sum_cn = 0.0
        for a in range(8):
            diff= abs(pixel_values[a] - pixel_values[a+1])
            sum_cn = sum_cn + diff
        return sum_cn // 2
    return 2.0

def remove_boundary_case(original_img,thin_img,minutae):
    minutiae_segment_mask = np.ones(original_img.shape)
    gloabl_var = np.var(thin_img)*0.1
    x= original_img.shape[0]
    y= original_img.shape[1]
    kernel_open_close = cv.getStructuringElement(cv.MORPH_RECT,(2*block_size, 2*block_size))
    for i in range(0, x, block_size):
        for j in range(0, y, block_size):
            end_i = min(x, i+block_size)
            end_j = min(y, j+block_size)
            value= thin_img[i: end_i, j: end_j]
            local_var = np.var(value)
            if local_var <= gloabl_var:
                minutiae_segment_mask[i: end_i, j: end_j] = 0.0
    print("Dilation")
    minutiae_segment_mask = cv.morphologyEx(minutiae_segment_mask, cv.MORPH_CLOSE, kernel_open_close)
    print("Erosion")
    minutiae_segment_mask = cv.morphologyEx(minutiae_segment_mask, cv.MORPH_OPEN, kernel_open_close)
    for i in range(0, x, block_size):
        end_i = min(x, i+block_size)
        minutiae_segment_mask[i: end_i, 0: block_size] = 0.0
        minutiae_segment_mask[i: end_i,y-block_size:y] = 0.0
    for j in range(0, y, block_size):
        end_j = min(y, i+block_size)
        minutiae_segment_mask[0: block_size, j: end_j] = 0.0
        minutiae_segment_mask[x-block_size: x, j:end_j] = 0.0
    minutae_new= new(minutiae_segment_mask,minutae)
    
    return minutae_new
def new(minutiae_segment_mask,minutae):
    new_minutiae = {}
    count=0
    for (x, y) in minutae:

        neighbourhood = [(0, 1), (0, -1), (0, 0), (1, 0), (-1, 0)]
        count=count+1
        to_append = True
        
        num=0
        for direction_x, direction_y in neighbourhood:
            x_= direction_x*block_size
            y_=direction_y*block_size
            try:
                if minutiae_segment_mask[x + x_, y + y_] == 0.0:
                    num=num+1
                    to_append = False
                    break
            except IndexError:
                num=num-1
                to_append = False
                break
            if to_append:
                new_minutiae[(x, y)] = minutae[(x, y)]
    minutae = new_minutiae
    return minutae

def draw_minutae(minutae,minutiae_img):
    bif=0
    ending=0
    for (x, y) in minutae:
        c_n, _ = minutae[(x, y)]
        
        if c_n == 3:
            bif=bif+1
            minutiae_img =cv.circle(minutiae_img, (y,x), radius=3, color=(0, 255, 0), thickness=1)
        if c_n == 1:
            ending=ending+1
            minutiae_img =cv.circle(minutiae_img, (y,x), radius=3, color=(0, 0, 255), thickness=1)
            
    return minutiae_img
def euclidean_distance(x1, y1, x2, y2):
    para1= x2-x1
    para2= y2-y1
    para3= para1**2 + para2**2
    return sqrt(para3)

def custom_round(x, base=5):
    denom= round(x/base)
    value= base * denom
    return value

def alignment(minutae_1,minutae_2):
    accumulator = {}
    for (xt, yt), (_, theta_t) in minutae_2.items():
        for (xq, yq), (_, theta_q) in minutae_1.items():
            d_theta = theta_t - theta_q
            parameter2= np.pi - d_theta
			# d_theta = abs(theta_t - theta_q)
            d_theta = min(d_theta, 2*parameter2)
            grad_x= math.cos(d_theta)
            grad_y= math.sin(d_theta)
            d_x = xt - xq*grad_x + yq*grad_y
            d_y = yt - xq*grad_y - yq*grad_x
            one= custom_round(180*d_theta/np.pi, 5)
            two= custom_round(d_x,block_size//4)
            three= custom_round(d_y,block_size//4)
            conf =one,two,three
            if conf not in accumulator:
                accumulator[conf] = 1
            else:
                accumulator[conf] += 1
    
    (theta, x, y) = max(accumulator, key=accumulator.get)
    theta_deg= np.pi*theta/180
    return theta_deg, x, y

def pair(minutae_1,minutae_2, transform_config):
    flag_q = np.zeros(len(minutae_1),)
    flag_t = np.zeros(len(minutae_2),)
    #print(len(minutae_1),len(minutae_2))
    count_matched = 0
    matched_minutiae = []
    angle_thresh = 10 * np.pi / 180
    distance_thresh =block_size/2
    ht_theta, ht_x, ht_y = transform_config
    i = 0
    ending=0
    bif=0
    for (xt, yt), (type1, theta_t) in minutae_2.items():
        j = 0
        for (xq, yq), (type2, theta_q) in minutae_1.items():
            d_theta = theta_t - theta_q - ht_theta
            parameter2= 2*np.pi - d_theta
            d_theta = min(d_theta,parameter2)
            grad_x= math.cos(ht_theta)
            grad_y= math.sin(ht_theta)
            d_x = xt - xq*grad_x + yq*grad_y - ht_x
            d_y = yt - xq*grad_y - yq*grad_x - ht_y
            c1=flag_t[i]
            c2=flag_q[j]
            c3= abs(d_theta) <= abs(angle_thresh)
            c4= euclidean_distance(0, 0, d_x, d_y) <= distance_thresh
            if  c1== 0.0 and c2 == 0.0 and c3 and c4:
                
                flag_t[i] = 1.0
                flag_t[i] = 1.0
                count_matched += 1
                if(type1==1 and type2==1):
                    ending=ending+1
                else:
                    bif=bif+1
                matched_minutiae.append(((xt, yt), (xq, yq)))
                

            j =j+ 1
        i =i+1
    return count_matched, i
def filename_func(add):
    head, tail = os.path.split(add)
    return tail
def split_string(string_value):
    split_string = string_value.split("_", 1)
    substring = split_string[0]
    return substring
def split_string_dot(string_value):
    split_string = string_value.split(".", 1)
    substring = split_string[0]
    return substring

def threshold_frr(score_list):
    frr_list = []
    threshold = []
    print("calculating FRR List")
    for i in range(100):
        num = 0
        for x in score_list:
            if x<i:
                num+=1
        threshold.append(i)
        frr_list.append(num)
        
    frr_list = np.array(frr_list)
    print('FRR: ',frr_list)
    print('ther: ',threshold)
    print('-----------------------------------------------------------')

if __name__ == "__main__":
 
 directory ="/Users/kashishjain/Desktop/Sem-II/Biometric Sec - SIL775/Assignments/A1_Biometric/Fingerprint-Verification-System-main/testcases/DB1_B"
 files = Path(directory).glob('*')
 score_list=[]
 imposter_list=[]
 for file in files:
    print(file)
    #image_path="/Users/kashishjain/Desktop/Sem-II/Biometric Sec - SIL775/Assignments/A1_Biometric/Fingerprint-Verification-System-main/testcases/104_1.tif"
    #image_path= f1
    image_path1="/Users/kashishjain/Desktop/Sem-II/Biometric Sec - SIL775/Assignments/A1_Biometric/Fingerprint-Verification-System-main/testcases/DB1_B/107_1.tif"
    p1= filename_func(image_path1)
    s1=split_string(p1)
    d1=split_string_dot(p1)
    image_one = cv.imread(image_path1 ,cv.IMREAD_GRAYSCALE )
    #interactive_display("Fingerprint 1",image_one)
    image_path2=str(file)
    image_two = cv.imread(image_path2 ,cv.IMREAD_GRAYSCALE )
    #interactive_display("Fingerprint 2",image_two)
    p2= filename_func(image_path2)
    s2=split_string(p2)
    d2=split_string_dot(p2)
    flag=0
    if(s1!=s2):
        print("Not Equal")
        continue
    print(s1,s2,flag)
    #Calculate Segment Mask
    seg_mask_one = seg_mask(image_one)
    #print(seg_mask_one)

    seg_mask_two = seg_mask(image_two)
    #print(seg_mask_two)
    # Apply seg mask
    seg_image_one = apply_mask(image_one,seg_mask_one)
    seg_image_two = apply_mask(image_two,seg_mask_two)

    #Apply normalisation
    nor_image_one= perform_normalisation(seg_image_one)
    nor_image_two= perform_normalisation(seg_image_two)

    #interactive_display("Fingerprint 1- Normalized",nor_image_one)
    #interactive_display("Fingerprint 2- Normalized",nor_image_two)
    
    #Image enchancement
    image_enhanced_one = fingerprint_enhancer.enhance_Fingerprint(nor_image_one)
    image_enhanced_two = fingerprint_enhancer.enhance_Fingerprint(nor_image_two)
    #interactive_display("Fingerprint 1- Enchanced",image_enhanced_one)
    #interactive_display("Fingerprint 2- Enchanced",image_enhanced_two)

    # Ridge Orientation'
    orientation_map_one = ridge_orientation_calculation(nor_image_one,image_one)
    orientation_map_two =ridge_orientation_calculation(nor_image_two,image_two)

    thinned_image_one = np.where(skeletonize(image_enhanced_one/255), 0.0, 1.0)
    thinned_image_two = np.where(skeletonize(image_enhanced_two/255), 0.0, 1.0)

    minutiae_one,minutiae_img_one = calculate_minutae(thinned_image_one,image_one,orientation_map_one)
    minutiae_two,minutiae_img_two = calculate_minutae(thinned_image_two,image_two,orientation_map_two)
    #print("M1: ",minutiae_one.items())
    #print("M2: ",minutiae_two)
    minutiae_one= remove_boundary_case(image_one,thinned_image_one,minutiae_one)
    minutiae_two =remove_boundary_case(image_two,thinned_image_two,minutiae_two)
    for (x, y) in minutiae_one:
        c_n, _ = minutiae_one[(x, y)]
        if c_n == 3:
            minutiae_img_one =cv.circle(minutiae_img_one, (y,x), radius=3, color=(0, 255, 0), thickness=1)
        if c_n == 1:
            minutiae_img_one =cv.circle(minutiae_img_one, (y,x), radius=3, color=(0, 0, 255), thickness=1)

    for (x, y) in minutiae_two:
        c_n, _ = minutiae_two[(x, y)]
        if c_n == 3:
            minutiae_img_two =cv.circle(minutiae_img_two, (y,x), radius=3, color=(0, 255, 0), thickness=1)
        if c_n == 1:
            minutiae_img_two =cv.circle(minutiae_img_two, (y,x), radius=3, color=(0, 0, 255), thickness=1)
            
    m_image_1= minutiae_img_one
    m_image_2= minutiae_img_two

    #interactive_display("Fingerprint 1- Minutae",m_image_1)
    #interactive_display("Fingerprint 2- Minutae",m_image_2)
    
    transform_config=alignment(minutiae_one,minutiae_two)
    match , i =pair(minutiae_one,minutiae_two, transform_config)
    score= min(match,100)
    #imposter_list.append(score)
    score_list.append(match)
    print(score_list)
    #print("{:.3f} percentage of minutiae matched".format(100*match/min(len(minutiae_one), len(minutiae_two))))
    # threshold value : 
 threshold_frr(score_list)