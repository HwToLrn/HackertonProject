import threading as trd
import time

def CurrentTime(term: int):
    count = 0
    while True:
        now = time.localtime()
        print(trd.currentThread().getName(), f'{now.tm_min}:{now.tm_sec}')
        if count == 3:
            thread2 = trd.Thread(target=CountDown, name='Child', args=(5, ))
            thread2.daemon = True
            if not thread2.is_alive():
                thread2.start()
            else:
                print(thread2.getName(), 'already exists')
            count = 0
        count += 1
        time.sleep(term)

# num만큼 counting하고 종료
def CountDown(num: int):
    for n in range(num):
        print(trd.currentThread().getName(), ' Count : ', n+1)
        time.sleep(0.4)

def main():
    thread1 = trd.Thread(target=CurrentTime, name='Parent', args=(2, ))
    thread1.start()

if __name__ == '__main__':
    main()