import os
import cv2
import numpy as np

def ADD_BLACK(img,TYPE=None):
    if TYPE=='RIGHT':
        canvas_black = np.zeros((100,50,3),dtype='uint8')
        img[0:100,50:100] = canvas_black
        print('Ture')
    elif TYPE=='LEFT':
        canvas_black = np.zeros((100,50,3),dtype='uint8')
        img[0:100,0:50] = canvas_black
        print('Ture')
    return img




#def main():
    

if __name__=='__main__':
    A_arr=[1,2,3]
    B_arr = [9,8,7]
    for i in range(len(B_arr)):
        A_arr.append(B_arr[i])
    print(A_arr)
    Path_input = r'F:/#DataSet/#FER/RAF-DB/basic/Image/aligned/'
    Path_output= r'F:/#DataSet/#FER/RAF-DB/basic/Image/Right_Occ/'
    count=0
    for root, _, basenames in os.walk(Path_input):
        for basename in basenames:
            if(basename.split('.')[1]=='jpg'):
                count+=1
                print('[INFO] processing.. %d '%(count),basename)
                img = cv2.imread(Path_input+basename)
                img_black = ADD_BLACK(img,TYPE='RIGHT')
                #cv2.imwrite(Path_output+basename,img_black,[cv2.IMWRITE_JPEG_QUALITY, 100])