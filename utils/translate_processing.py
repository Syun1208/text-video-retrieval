import googletrans
import langdetect
import translate
from text_generation import Client

class Translation:
    def __init__(self, from_lang='vi', to_lang='en', mode='googletrans'):
        # The class Translation is a wrapper for the two translation libraries, googletrans and translate. 
        self.__mode = mode
        self.__from_lang = from_lang
        self.__to_lang = to_lang

        if mode in 'googletrans':
            self.translator = googletrans.Translator()
        elif mode in 'translate':
            self.translator = translate.Translator(from_lang=from_lang,to_lang=to_lang)

    def preprocessing(self, text):
        """
        It takes a string as input, and returns a string with all the letters in lowercase
        :param text: The text to be processed
        :return: The text is being returned in lowercase.
        """
        return text.lower()

    def __call__(self, text):
        """
        The function takes in a text and preprocesses it before translation
        :param text: The text to be translated
        :return: The translated text.
        """
        text = self.preprocessing(text)
        return self.translator.translate(text) if self.__mode in 'translate' \
                else self.translator.translate(text, dest=self.__to_lang).text
    
if __name__ == "__main__":
    client = Client("http://127.0.0.1:8080")
    translator = translate.Translator(from_lang='vi',to_lang='en')
    print(client.generate(translator.translate("Lòng gà nướng là gì ?"), max_new_tokens=20).generated_text)

    text = ""
    for response in client.generate_stream(translator.translate("Lòng gà nướng là gì ?"), max_new_tokens=20):
        if not response.token.special:
            text += response.token.text
    print(text)