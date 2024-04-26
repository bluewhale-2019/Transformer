import spacy

class Tokenizer:
    def __init__(self) -> None:
        self.spacy_encoder = spacy.load('en_core_web_sm')
        self.spacy_deconder = spacy.load('de_core_news_sm')
        
    def tokenizer_encoder(self, text):
        return [token.text for token in self.spacy_encoder.tokenizer(text)]
    
    def tokenizer_decoder(self, text):
        return [token.text for token in self.spacy_deconder.tokenizer(text)]
    