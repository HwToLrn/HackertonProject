import gtts

def main():
    # {'af': 'Afrikaans', 'sq': 'Albanian', 'ar': 'Arabic', 'hy': 'Armenian', 'bn': 'Bengali', 'bs': 'Bosnian',
    #  'ca': 'Catalan', 'hr': 'Croatian', 'cs': 'Czech', 'da': 'Danish', 'nl': 'Dutch', 'en': 'English',
    #  'eo': 'Esperanto', 'et': 'Estonian', 'tl': 'Filipino', 'fi': 'Finnish', 'fr': 'French', 'de': 'German',
    #  'el': 'Greek', 'gu': 'Gujarati', 'hi': 'Hindi', 'hu': 'Hungarian', 'is': 'Icelandic', 'id': 'Indonesian',
    #  'it': 'Italian', 'ja': 'Japanese', 'jw': 'Javanese', 'kn': 'Kannada', 'km': 'Khmer', 'ko': 'Korean', 'la': 'Latin',
    #  'lv': 'Latvian', 'mk': 'Macedonian', 'ml': 'Malayalam', 'mr': 'Marathi', 'my': 'Myanmar (Burmese)',
    #  'ne': 'Nepali', 'no': 'Norwegian', 'pl': 'Polish', 'pt': 'Portuguese',
    #  'ro': 'Romanian', 'ru': 'Russian', 'sr': 'Serbian', 'si': 'Sinhala', 'sk': 'Slovak', 'es': 'Spanish',
    #  'su': 'Sundanese', 'sw': 'Swahili', 'sv': 'Swedish', 'ta': 'Tamil', 'te': 'Telugu', 'th': 'Thai', 'tr': 'Turkish',
    #  'uk': 'Ukrainian', 'ur': 'Urdu', 'vi': 'Vietnamese', 'cy': 'Welsh', 'zh-cn': 'Chinese (Mandarin/China)',
    #  'zh-tw': 'Chinese (Mandarin/Taiwan)', 'en-us': 'English (US)', 'en-ca': 'English (Canada)',
    #  'en-uk': 'English (UK)', 'en-gb': 'English (UK)', 'en-au': 'English (Australia)', 'en-gh': 'English (Ghana)',
    #  'en-in': 'English (India)', 'en-ie': 'English (Ireland)', 'en-nz': 'English (New Zealand)',
    #  'en-ng': 'English (Nigeria)', 'en-ph': 'English (Philippines)', 'en-za': 'English (South Africa)',
    #  'en-tz': 'English (Tanzania)', 'fr-ca': 'French (Canada)', 'fr-fr': 'French (France)',
    #  'pt-br': 'Portuguese (Brazil)', 'pt-pt': 'Portuguese (Portugal)', 'es-es': 'Spanish (Spain)',
    #  'es-us': 'Spanish (United States)'}

    langs = ['ko', 'en-us', 'ja']
    filenames = ['../SavedSpeech/test_ko.mp3', '../SavedSpeech/test_en-us.mp3', '../SavedSpeech/test_ja.mp3']
    for lang, fn in zip(langs, filenames):
        tts = gtts.gTTS("셋", lang=lang)
        tts.save(fn)

if __name__ == '__main__':
    main()