
def clean_text(text): #ardisik ayni 3 harfi teke düsürür
    import re
    cleaned_text = re.sub(r'(.)\1{3,}', r'\1', text)  
    return cleaned_text

def detect_smiley(text): 
    smileys = {
        ':)': 'smiley_positive ',
        ':-)': 'smiley_positive ',
        ':(': 'smiley_negative ',
        ':-(': 'smiley_negative ',
        ':D': 'smiley_very_positive ',
        ':-D': 'smiley_very_positive ',
        ':|': 'smiley_neutral '
    }

    for smiley, replacement in smileys.items():
        text = text.replace(smiley, replacement)
    
    return text

def detect_language(text):
    from langdetect import detect
    language = detect(text)
    return language

def translate_to_english(text):
    from googletrans import Translator
    translator = Translator()
    translation = translator.translate(text, dest='en')
    return translation.text

def safe_detect(text):
    from langdetect import detect
    try:
        # Boş metin veya çok kısa metin kontrolü
        if text.strip() == '' or len(text) < 3:
            return 'unknown'  # Dil belirlenemeyen metinler için 'unknown' dönebiliriz
        return detect(text)
    except:
        return 'unknown' 
    

def extract_emojis(text):
    import demoji
    emoji_dict = demoji.findall(text)
    if bool(emoji_dict):
        return emoji_dict
    else:
        return False
    
'''
def replace_emoji(text):
    emojis = {
        '☹' : 'sad',
        '☺' : 'smile',
        '♥' : 'love',
        '✔️' : 'check_mark',
        '🎉' : 'celebration',
        '👌' : 'okay',
        '👌🏻' : 'okay',
        '👌🏼' : 'okay',
        '👌🏽' : 'okay',
        '👍' : 'thumbs_up',
        '👍🏻' : 'thumbs_up',
        '👍🏼' : 'thumbs_up',
        '👎' : 'thumbs_down',
        '👎🏻' : 'thumbs_down',
        '👎🏼' : 'thumbs_down',
        '👏' : 'clap',
        '👏🏻' : 'clap',
        '💓' : 'heartbeat',
        '💕' : 'two_hearts',
        '💙' : 'blue_heart',
        '💚' : 'green_heart',
        '💛' : 'yellow_heart',
        '💜' : 'purple_heart',
        '🧡' : 'orange_heart',
        '🖤' : 'black_heart',
        '💯' : 'perfect',
        '😀' : 'grin',
        '😁' : 'grinning',
        '😂' : 'laughing',
        '😃' : 'smile',
        '😄' : 'big_smile',
        '😅' : 'sweat_smile',
        '😆' : 'laugh',
        '😇' : 'angel',
        '😊' : 'blush',
        '😋' : 'yummy',
        '😍' : 'heart_eyes',
        '😈' : 'devilish',
        '😉' : 'wink',
        '😊' : 'blush',
        '😋' : 'yummy',
        '😌' : 'relieved',
        '😍' : 'heart_eyes',
        '😎' : 'sunglasses',
        '😐' : 'neutral',
        '😑' : 'expressionless',
        '😒' : 'unamused',
        '😓' : 'sweat',
        '😔' : 'pensive',
        '😕' : 'confused',
        '😖' : 'disappointed',
        '😘' : 'blowing_kiss',
        '😚' : 'kiss',
        '😛' : 'tongue',
        '😜' : 'winking_tongue',
        '😝' : 'crazy_tongue',
        '😠' : 'angry',
        '😡' : 'mad',
        '😢' : 'cry',
        '😣' : 'persevere',
        '😤' : 'triumph',
        '😥' : 'sad',
        '😩' : 'weary',
        '😪' : 'sleepy',
        '😬' : 'grimace',
        '😭' : 'sob',
        '😯' : 'surprised',
        '😲' : 'astonished',
        '😳' : 'flushed',
        '😷' : 'mask',
        '🙁' : 'frowning',
        '🙂' : 'slightly_smiling',
        '🙃' : 'upside_down',
        '🙄' : 'rolling_eyes',
        '🙏' : 'pray',
        '🙏🏻' : 'pray',
        '🙏🏽' : 'pray',
        '🤐' : 'zipper_mouth',
        '🤔' : 'thinking',
        '🤗' : 'hugging',
        '🤢' : 'nauseated',
        '🤣' : 'rofl',
        '🤤' : 'drooling',
        '🤨' : 'raised_eyebrow',
        '🤩' : 'star_struck',
        '🤪' : 'crazy_face',
        '🤬' : 'cursing',
        '🤭' : 'hand_over_mouth',
        '🤮' : 'vomit',
        '🥰' : 'smiling_hearts',
        '🥳' : 'partying',
        '🥵' : 'hot_face',
    }

    for emoji_char, keyword in emojis.items():
        text = text.replace(emoji_char, keyword)
    return text
'''
    
def replace_emoji(text):
    import emoji
    return emoji.demojize(text)


