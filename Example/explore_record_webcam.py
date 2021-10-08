import cv2


def main():
    capture = cv2.VideoCapture(0)

    OUTPUT_FILENAME = '../SavedVideo/SavedVideo.avi'
    width = int(capture.get(3))
    height = int(capture.get(4))

    # 코덱정보를 나타냄 아래의 두줄과 같이 사용할 수 있음. 둘 중 어느것을 쓰든 상관없음.
    # 여러가지의 코덱종류가 있지만 윈도우라면 DIVX 를 사용
    # fourcc = cv2.VideoWriter_fourcc('D','I','V','X')
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') # 코덱 정보

    # 비디오 저장을 위한 객체를 생성
    output = cv2.VideoWriter(filename=OUTPUT_FILENAME, fourcc=fourcc, fps=20.0, frameSize=(width, height))

    while capture.isOpened():
        existframe, frame = capture.read()

        if existframe:
            # 세로(vertical) 기준 : 0 / 가로(horizontal) 기준 : 1
            frame = cv2.flip(src=frame, flipCode=1)
            cv2.imshow(winname='MyWindow', mat=frame)
            output.write(image=frame)

        # ESC key for closing the window
        if cv2.waitKey(1) & 0xFF == 27:
            break

    capture.release()
    output.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()