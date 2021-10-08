import cv2

def main():
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        existframe, frame = capture.read()

        if existframe:
            # 세로(vertical) 기준 : 0 / 가로(horizontal) 기준 : 1
            frame = cv2.flip(src=frame, flipCode=1)
            cv2.imshow(winname='MyWindow', mat=frame)

        key = cv2.waitKey(1)
        # ESC key for closing the window
        if key & 0xFF == 27:
            break
        elif key == ord('p'): # ord() : 문자열을 아스키코드로 변환
            cv2.waitKey(-1) # wait until any key is pressed

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()