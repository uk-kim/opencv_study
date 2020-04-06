import os
import numpy as np
import cv2


def rotate_image(src):
    dst = src.transpose([1, 0, 2])
    dst = cv2.flip(dst, 1)
    return dst

def get_mask_from_hsv(src, lower, upper):
    lower_np = np.array(lower)
    upper_np = np.array(upper)

    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_np, upper_np)
    #mask = cv2.inRange(hsv, lower, upper)
    return mask

def calc_centroid_and_std(src):
    # Var(X) = E(X^2) - {E(X)}^2
    # Std(X) = sqrt(Var(X))
    rows = src.shape[0]
    cols = src.shape[1]

    sum_x = sum_y = 0
    sum_xx = sum_yy = 0
    
    # To find centroid, select indexes with non-zero values
    idxs = np.argwhere(src > 0)
    idxs_sq = np.square(idxs)
    
    # To calc avg and var, calc sum(X) and sum(X^2)
    sum_xy = np.sum(idxs, 0)
    sum_xxyy = np.sum(idxs_sq, 0)
    n_pixel = len(idxs)
   
    # Espectation value of non-zero pixels coordinate indexes
    E_xx = sum_xxyy / n_pixel
    E_x  = sum_xy / n_pixel
   
    # avg_xy : centroid of non-zero pixels
    # var_xy : variance of non-zero pixels
    # std_xy : variance of non-zero pixels
    avg_xy = E_x
    var_xy = E_xx - np.square(E_x)
    std_xy = np.sqrt(var_xy)

    return avg_xy, std_xy
    

def main():
    video_file = './tracking_2.mp4'
    cap = cv2.VideoCapture(video_file)

    while True:
        if(cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            print(" The end of this video")
            print(" Press any key to terminate this program")
            cv2.waitKey(5000)
            break
        
        ret, frame = cap.read()
        if not ret:
            print(" Something wrong with video capture. Maybe caused by video file..")
            print(" Press any key to terminate this program.")
            cv2.waitKey(5000)
            break

        frame = rotate_image(frame)

        # Get HSV mask
        lower = [0, 120 + 50, 70]
        upper = [5, 255, 255]
        mask1 = get_mask_from_hsv(frame, lower, upper)

        lower = [175, 120 + 50, 70]
        upper = [180, 255, 255]
        mask2 = get_mask_from_hsv(frame, lower, upper)

        mask = mask1 + mask2
        #mask1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        
        # To filt noise pixel, use morphological operation with openning and closing
        mask_fg = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

        #creating an inverted mask to segment out the cloth from the frame
        mask_bg = cv2.bitwise_not(mask_fg)
        
        frame_fg = cv2.bitwise_and(frame, frame, mask=mask_fg)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_bg)
        
        centroid, std = calc_centroid_and_std(mask_fg)
        cx = int(centroid[1])
        cy = int(centroid[0])
        x_sig = int(std[1] * 2)
        y_sig = int(std[0] * 2)

        frame = cv2.rectangle(frame, (cx-x_sig, cy-y_sig), (cx+x_sig, cy+y_sig), color=(0, 255, 0), thickness=3)

        frame = cv2.putText(frame, "Source Image (with red ball detect)", (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)
        frame_fg = cv2.putText(frame_fg, "Foreground Image (red ball)", (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)
        frame_bg = cv2.putText(frame_bg, "Background Image (red ball)", (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)
        frame = np.concatenate([frame, frame_fg, frame_bg], 1)
        cv2.imshow("video", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
