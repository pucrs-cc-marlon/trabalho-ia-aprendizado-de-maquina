# -.- encoding: utf-8 -.-

import re
import nltk
import pickle

from bs4 import BeautifulSoup

TESTE_DIR = 'CORPUS/TESTE/'
TREINO_DIR = 'CORPUS/TREINO/'
ARQUIVOS = ['CORPUS ESPACO DO TRABALHADOR.txt', 'CORPUS ESPORTES.txt',
            'CORPUS POLICIA.txt', 'CORPUS SEU PROBLEMA E NOSSO.txt']


class LimparArquivosTexto():

    def __init__(self):
        self.clean_text(ARQUIVOS)

    def clean_text(self, arquivos):
        for filename in arquivos:
            with open('CORPUS/{}'.format(filename), 'r') as arq:
                texto = ' '.join(arq.readlines())
                bes = BeautifulSoup(texto, "html.parser") # Removendo as marcações
                texto = bes.get_text() # Remove as Tags do texto
                texto = self.remove_emails(texto) # Remover os e-mails
                texto = self.remove_stopwords(texto) # Remover as stopwords
                texto = nltk.wordpunct_tokenize(texto) # Tokenizar as palavras
                pos_texto = self.pos_annotation(texto) # Adição da notação POS (part of speech)
                texto = nltk.Text(texto) # Texto nos padrões do NLTK


    def remove_emails(self, text):
        re_email = r'\S*@\S*\s?'
        pattern = re.compile(re_email)
        text = pattern.sub('', text)
        return text

    def remove_stopwords(self, text):
        stopwords = set(nltk.corpus.stopwords.words('portuguese'))
        words = [word for word in text if word not in stopwords]
        return words

    def pos_annotation(self, text):
        tagger = pickle.load(open("tagger.pkl"))
        portuguese_sent_tokenizer = nltk.data.load("tokenizers/punkt/portuguese.pickle")
        sentences = portuguese_sent_tokenizer.tokenize(text)
        tags = [tagger.tag(nltk.word_tokenize(sentence)) for sentence in sentences]
        return tags
