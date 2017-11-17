# -.- encoding: utf-8 -.-

import re
import nltk
import pickle

from bs4 import BeautifulSoup
from cogroo_interface import Cogroo

TESTE_DIR = 'CORPUS/TESTE/'
TREINO_DIR = 'CORPUS/TREINO/'
ARQUIVOS = ['CORPUS ESPACO DO TRABALHADOR.txt', ]
# 'CORPUS ESPORTES.txt', 'CORPUS POLICIA.txt', 'CORPUS SEU PROBLEMA E NOSSO.txt']


class LimparArquivosTexto:
    def __init__(self):
        self.generate_structured_files(ARQUIVOS)
        cogroo = Cogroo.Instance()

    def generate_structured_files(self, arquivos):
        for filename in arquivos:
            with open('CORPUS/{}'.format(filename), 'r') as arq:
                lines = arq.readlines()
                texto = ' '.join(lines)
                textos = self.split_texts(lines)
                textos = self.clean_texts(textos)
                # self.create_bag_of_words(texto=texto)

    def split_texts(self, lines):
        textos = {}
        actual_text = None
        for line in lines:
            if line.find('TEXTO') == 0:
                actual_text = line.replace('\n', '')
                textos[actual_text] = []
            else:
                if actual_text is not None:
                    textos[actual_text].append(line)
        return textos

    def clean_texts(self, textos):
        for texto in textos:
            t = textos[texto]

        return textos

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

    def lemmatize(self, text):
        pass

    def create_bag_of_words(self, texto):
        bes = BeautifulSoup(texto, "html.parser")  # Removendo as marcações
        texto = bes.get_text()  # Remove as Tags do texto
        texto = self.remove_emails(texto)  # Remover os e-mails
        texto = self.remove_stopwords(texto)  # Remover as stopwords
        texto = nltk.wordpunct_tokenize(texto)  # Tokenizar as palavras
        # pos_texto = self.pos_annotation(texto)  # Adição da notação POS (part of speech)
        texto = nltk.Text(texto)  # Texto nos padrões do NLTK


if __name__ == "__main__":
    clean_files = LimparArquivosTexto()
