import datetime
import time
from threading import Thread
import cv2
# from __future__ import print_function
# from imutils.video import WebcamVideoStream
# from imutils.video import FPS
import argparse
import imutils


class FPS:
    def __init__(self):
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        self._end = datetime.datetime.now()

    def update(self):
        self._numFrames += 1

    def elapsed(self):
        return (self._end - self._start).total_seconds()

    def fps(self):
        return self._numFrames / self.elapsed()


class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

def baselinefps():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num-frames", type=int, default=100,
                    help="# of frames to loop over for FPS test")
    ap.add_argument("-d", "--display", type=int, default=-1,
                    help="Whether or not frames should be displayed")
    args = vars(ap.parse_args())

    print("[INFO] sampling frames from webcam...")
    stream = cv2.VideoCapture(0)
    fps = FPS().start()

    # loop over some frames
    while fps._numFrames < args["num_frames"]:
        # grab the frame from the stream and resize it to have a maximum
        # width of 400 pixels
        (grabbed, frame) = stream.read()
        frame = imutils.resize(frame, width=400)
        # check to see if the frame should be displayed to our screen
        if args["display"] > 0:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
        # update the FPS counter
        fps.update()

    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed())) # 3.43
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps())) # 29.17

    stream.release()
    cv2.destroyAllWindows()


def comparefps():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num-frames", type=int, default=100,
                    help="# of frames to loop over for FPS test")
    ap.add_argument("-d", "--display", type=int, default=-1,
                    help="Whether or not frames should be displayed")
    args = vars(ap.parse_args())

    print("[INFO] sampling THREADED frames from webcam...")
    vs = WebcamVideoStream(src=0).start()
    fps = FPS().start()

    while fps._numFrames < args["num_frames"]:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        if args["display"] > 0:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

        fps.update()

    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed())) # 0.51
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps())) # about 200

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

def experience_fps():
    prev_time = 0
    FPS = 120
    video = cv2.VideoCapture(0)

    while True:
        ret, frame = video.read()
        current_time = time.time() - prev_time
        print(f'{current_time} / Set fps : {1./FPS}')
        if ret and (current_time > 1. / FPS):

            prev_time = time.time()

            cv2.imshow('VideoCapture', frame)

            if cv2.waitKey(1) > 0:
                break

def main():
    pass


if __name__ == '__main__':
	# main()
    # baselinefps()
    # comparefps()
    experience_fps()