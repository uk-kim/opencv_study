import cv2
import numpy as np

def onMouse(event, x, y, flags, param): # 아무스 콜백 함수 구현 ---①
    color = colors['black']
    if event == cv2.EVENT_MOUSEMOVE:
        color = param['colors']
        r = color['red']
        g = color['green']
        b = color['blue']
        white = color['white']
       
        disp_img = param['template'].copy()
        rows = param['img'].shape[0]
        cols = param['img'].shape[1]
       
	    if x < 0 or x >= cols or y < 0 or y >= rows:
		    continue

        row = param['img'][y,:]
        col = param['img'][:,x]
        
        for i in range(len(row[:-1])):
            _c1 = row[i]
            _c2 = row[i+1]
            
            cv2.line(disp_img, (i, rows+_c1[0]), (i+1, rows+_c2[0]), b)
            cv2.line(disp_img, (i, rows+_c1[1]), (i+1, rows+_c2[1]), g)
            cv2.line(disp_img, (i, rows+_c1[2]), (i+1, rows+_c2[2]), r)

        for i in range(len(col[:-1])):
            _c1 = col[i]
            _c2 = col[i+1]

            cv2.line(disp_img, (cols+_c1[0], i), (cols+_c2[0], i+1), b)
            cv2.line(disp_img, (cols+_c1[1], i), (cols+_c2[1], i+1), g)
            cv2.line(disp_img, (cols+_c1[2], i), (cols+_c2[2], i+1), r)

        cv2.imshow(param['title'], disp_img)
        cv2.waitKey(1)
        

if __name__ == "__main__":
    title="Image"
    colors = {'black':(0,0,0),
        'red' : (0,0,255),
        'blue':(255,0,0),
        'green': (0,255,0),
        'white': (255, 255, 255)} # 색상 미리 정의

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cnt = 0
        img = None
        while cnt < 100:
            ret, frame = cap.read()
            if ret:
                img = frame.copy()
                break
        cap.release()
    else:
        cap.release()
        raise "Capture open failed"
   
    img = cv2.resize(img, (int(img.shape[1]*0.7), int(img.shape[0]*0.7)))
    img_template = np.zeros((img.shape[0] + 255, img.shape[1] + 255, 3), dtype=np.uint8)
    img_template[:,:,:] = 255
    img_template[:img.shape[0], :img.shape[1],:] = img[:,:,:]

    cv2.imshow(title, img_template)                  # 백색 이미지 표시
    
    param = {
        'img': img.copy(),
        'template': img_template.copy(),
        'colors': colors,
        'title': title
    }
    cv2.setMouseCallback(title, onMouse, param)

    while True:
        if cv2.waitKey(0) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

