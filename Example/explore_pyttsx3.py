import pyttsx3
from typing import List

# Voice 추가하는 방법 Link
# https://puneet166.medium.com/how-to-added-more-speakers-and-voices-in-pyttsx3-offline-text-to-speech-812c83d14c13
def main():
    # OS platform마다 지원하는 driverName이 다르다. Window는 'sapi5'만 지원한다.
    engine = pyttsx3.init(driverName='sapi5')

    # Set speed of voice
    engine.setProperty(name='rate', value=170)
    # Set voice
    voices = engine.getProperty(name='voices')
    engine.setProperty(name='voice', value=voices[0].id) # 0:KO-KR / 1:EN-US

    count: List[str] = ['하나', '둘', '셋', '넷', '다섯']
    for str_num in count:
        engine.say(text=str_num)
        engine.runAndWait()


def test():
    engine = pyttsx3.init()  # object creation

    """ RATE"""
    rate = engine.getProperty('rate')  # getting details of current speaking rate
    print(rate)  # printing current voice rate
    engine.setProperty('rate', 125)  # setting up new voice rate

    """VOLUME"""
    volume = engine.getProperty('volume')  # getting to know current volume level (min=0 and max=1)
    print(volume)  # printing current volume level
    engine.setProperty('volume', 1.0)  # setting up volume level  between 0 and 1

    """VOICE"""
    voices = engine.getProperty('voices')  # getting details of current voice
    engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
    # engine.setProperty('voice', voices[1].id)  # changing index, changes voices. 1 for female

    engine.say("Hello World!")
    engine.say('My current speaking rate is ' + str(rate))
    engine.runAndWait()
    engine.stop()

if __name__ == '__main__':
    main()
    # test()