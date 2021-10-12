from typing import List


def main():
    data: List[int] = [1, 2, 3]
    try:
        print(data[4])
    except IndexError:
        print('수를 다 세었습니다.')



if __name__ == '__main__':
    main()