# -.- encoding: utf-8 -.-

import re
import nltk
import json
import random
import pickle

from bs4 import BeautifulSoup
from cogroo_interface import Cogroo

TESTE_DIR = 'CORPUS/TESTE/'
TREINO_DIR = 'CORPUS/TREINO/'
ARQUIVOS = ['CORPUS ESPACO DO TRABALHADOR.txt',
            'CORPUS ESPORTES.txt', 'CORPUS POLICIA.txt', 'CORPUS SEU PROBLEMA E NOSSO.txt']


class EstruturarTextos:
    def __init__(self):
        self.bag_of_words_treino = {'unigram': {}, 'bigram': {}, 'trigram': {}}
        self.bag_of_words_teste = {'unigram': {}, 'bigram': {}, 'trigram': {}}
        self.generate_structured_files(ARQUIVOS)
        self.create_unified_bag_of_words()

    def generate_structured_files(self, arquivos):
        for filename in arquivos:
            with open('CORPUS/{}'.format(filename), 'r') as arq:
                lines = arq.readlines()
                textos = self.split_texts(lines)
                textos = self.clean_texts(textos)

                len_size_treino = len(textos) * 0.8

                textos_treino = self.get_texts(textos, 0, int(len_size_treino))
                textos_teste = self.get_texts(textos, int(len_size_treino), len(textos))

                bow_treino_uni, bow_treino_bi, bow_treino_tri = self.create_bag_of_words(textos_treino)
                self.join_bag_of_words_treino(bow_treino_uni, 'unigram')
                self.join_bag_of_words_treino(bow_treino_bi, 'bigram')
                self.join_bag_of_words_treino(bow_treino_tri, 'trigram')

                bow_teste_uni, bow_teste_bi, bow_teste_tri = self.create_bag_of_words(textos_teste)
                self.join_bag_of_words_teste(bow_teste_uni, 'unigram')
                self.join_bag_of_words_teste(bow_teste_bi, 'bigram')
                self.join_bag_of_words_teste(bow_teste_tri, 'trigram')

                # Criando arquivos do corpus de TREINO
                with open('CORPUS/TREINO/{}'.format(filename), 'w') as new_arq:
                    to_json = json.dumps(textos_treino, ensure_ascii=False, indent=2)
                    new_arq.write(to_json)

                with open('CORPUS/TREINO/BoW/uni_{}'.format(filename), 'w') as new_arq:
                    words = [(k, bow_treino_uni[k]) for k in sorted(bow_treino_uni,
                                                                    key=bow_treino_uni.get, reverse=True)]
                    for k, v in words:
                        new_arq.write("{} - {}\n".format(k, v))

                with open('CORPUS/TREINO/BoW/uni_bi_{}'.format(filename), 'w') as new_arq:
                    bow_treino_uni_bi = {**bow_treino_uni, **bow_treino_bi}
                    words = [(k, bow_treino_uni_bi[k]) for k in sorted(bow_treino_uni_bi,
                                                                       key=bow_treino_uni_bi.get, reverse=True)]
                    for k, v in words:
                        new_arq.write("{} - {}\n".format(k, v))

                with open('CORPUS/TREINO/BoW/uni_bi_tri_{}'.format(filename), 'w') as new_arq:
                    bow_treino_uni_bi_tri = {**bow_treino_uni, **bow_treino_bi, **bow_treino_tri}
                    words = [(k, bow_treino_uni_bi_tri[k]) for k in sorted(bow_treino_uni_bi_tri,
                                                                           key=bow_treino_uni_bi_tri.get, reverse=True)]
                    for k, v in words:
                        new_arq.write("{} - {}\n".format(k, v))

                # Criando arquivos do corpus de TESTE
                with open('CORPUS/TESTE/{}'.format(filename), 'w') as new_arq:
                    to_json = json.dumps(textos_teste, ensure_ascii=False, indent=2)
                    new_arq.write(to_json)

                with open('CORPUS/TESTE/BoW/uni_{}'.format(filename), 'w') as new_arq:
                    words = [(k, bow_teste_uni[k]) for k in sorted(bow_teste_uni,
                                                                   key=bow_teste_uni.get, reverse=True)]
                    for k, v in words:
                        new_arq.write("{} - {}\n".format(k, v))

                with open('CORPUS/TESTE/BoW/uni_bi_{}'.format(filename), 'w') as new_arq:
                    bow_teste_uni_bi = {**bow_teste_uni, **bow_teste_bi}
                    words = [(k, bow_teste_uni_bi[k]) for k in sorted(bow_teste_uni_bi,
                                                                      key=bow_teste_uni_bi.get, reverse=True)]
                    for k, v in words:
                        new_arq.write("{} - {}\n".format(k, v))

                with open('CORPUS/TESTE/BoW/uni_bi_tri_{}'.format(filename), 'w') as new_arq:
                    bow_teste_uni_bi_tri = {**bow_teste_uni, **bow_teste_bi, **bow_teste_tri}
                    words = [(k, bow_teste_uni_bi_tri[k]) for k in sorted(bow_teste_uni_bi_tri,
                                                                          key=bow_teste_uni_bi_tri.get, reverse=True)]
                    for k, v in words:
                        new_arq.write("{} - {}\n".format(k, v))

    def create_unified_bag_of_words(self):
        with open('CORPUS/TREINO/BoW/All/all_unigram.txt', 'w') as new_arq:
            words = [(k, self.bag_of_words_treino['unigram'][k]) for k in sorted(self.bag_of_words_treino['unigram'],
                                                                                 key=self.bag_of_words_treino[
                                                                                     'unigram'].get, reverse=True)]
            for k, v in words:
                new_arq.write("{} - {}\n".format(k, v))

        with open('CORPUS/TREINO/BoW/All/all_unigram_bigram.txt', 'w') as new_arq:
            bow_treino_uni_bi = {**self.bag_of_words_treino['unigram'], **self.bag_of_words_treino['bigram']}
            words = [(k, bow_treino_uni_bi[k]) for k in sorted(bow_treino_uni_bi,
                                                               key=bow_treino_uni_bi.get, reverse=True)]
            for k, v in words:
                new_arq.write("{} - {}\n".format(k, v))

        with open('CORPUS/TREINO/BoW/All/all_unigram_bigram_trigram.txt', 'w') as new_arq:
            bow_treino_uni_bi_tri = {**self.bag_of_words_treino['unigram'], **self.bag_of_words_treino['bigram'],
                                     **self.bag_of_words_treino['trigram']}
            words = [(k, bow_treino_uni_bi_tri[k]) for k in sorted(bow_treino_uni_bi_tri,
                                                                   key=bow_treino_uni_bi_tri.get, reverse=True)]
            for k, v in words:
                new_arq.write("{} - {}\n".format(k, v))

        with open('CORPUS/TESTE/BoW/All/all_unigram.txt', 'w') as new_arq:
            words = [(k, self.bag_of_words_teste['unigram'][k]) for k in sorted(self.bag_of_words_teste['unigram'],
                                                                                key=self.bag_of_words_teste[
                                                                                    'unigram'].get, reverse=True)]
            for k, v in words:
                new_arq.write("{} - {}\n".format(k, v))

        with open('CORPUS/TESTE/BoW/All/all_unigram_bigram.txt', 'w') as new_arq:
            bow_teste_uni_bi = {**self.bag_of_words_teste['unigram'], **self.bag_of_words_teste['bigram']}
            words = [(k, bow_teste_uni_bi[k]) for k in sorted(bow_teste_uni_bi,
                                                              key=bow_teste_uni_bi.get, reverse=True)]
            for k, v in words:
                new_arq.write("{} - {}\n".format(k, v))

        with open('CORPUS/TESTE/BoW/All/all_unigram_bigram_trigram.txt', 'w') as new_arq:
            bow_teste_uni_bi_tri = {**self.bag_of_words_teste['unigram'], **self.bag_of_words_teste['bigram'],
                                    **self.bag_of_words_teste['trigram']}
            words = [(k, bow_teste_uni_bi_tri[k]) for k in sorted(bow_teste_uni_bi_tri,
                                                                  key=bow_teste_uni_bi_tri.get, reverse=True)]
            for k, v in words:
                new_arq.write("{} - {}\n".format(k, v))

    def join_bag_of_words_teste(self, bag_of_words, tipo):
        for i in bag_of_words:
            if i in self.bag_of_words_teste:
                self.bag_of_words_teste[tipo][i] += bag_of_words[i]
            else:
                self.bag_of_words_teste[tipo][i] = bag_of_words[i]

    def join_bag_of_words_treino(self, bag_of_words, tipo):
        for i in bag_of_words:
            if i in self.bag_of_words_treino:
                self.bag_of_words_treino[tipo][i] += bag_of_words[i]
            else:
                self.bag_of_words_treino[tipo][i] = bag_of_words[i]

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
        for titulo in textos:
            texto = textos[titulo]
            texto = " ".join(texto)
            bes = BeautifulSoup(texto, "html.parser")
            texto = bes.get_text()  # Remove as Tags do texto
            texto = self.remove_emails(texto)  # Remover os e-mails
            texto = self.lemmatize(texto)
            tags = self.pos_annotation(texto)
            unigram, bigrams, trigrams = self.create_ngrams(tags)
            textos[titulo] = {'texto': texto, 'pos': tags, 'ngrams': {'unigram': unigram,
                                                                      'bigrams': bigrams, 'trigrams': trigrams}}
        return textos

    def create_ngrams(self, tags):
        unigram = []
        bigrams = []
        trigrams = []

        # Unigram
        for words in tags:
            tmp_words = []
            for w in words:
                if w[1] == 'VERB' or w[1] == 'ADJ' or w[1] == 'ADV' or w[1] == 'NOUN':
                    tmp_words.append(w[0])
            for u in nltk.ngrams(tmp_words, 1):
                unigram.append(u)

        # Bigrams
        for words in tags:
            tmp_words = []
            for w in words:
                if w[1] == 'VERB' or w[1] == 'ADP' or w[1] == 'ADJ' or w[1] == 'ADV' or w[1] == 'NOUN':
                    tmp_words.append(w[0])
            for u in nltk.ngrams(tmp_words, 2):
                bigrams.append(u)

        # Trigrams
        for words in tags:
            tmp_words = []
            for w in words:
                if w[1] == 'VERB' or w[1] == 'ADP' or w[1] == 'ADJ' or w[1] == 'ADV' or w[1] == 'NOUN':
                    tmp_words.append(w[0])
            for u in nltk.ngrams(tmp_words, 3):
                trigrams.append(u)

        return unigram, bigrams, trigrams

    def get_random_texts(self, texts, percent):
        # Necessita de melhorias
        len_size = len(texts) * percent
        new_texts = {}
        for i in range(0, int(len_size)):
            text = random.choice(list(texts.keys()))
            if text not in new_texts:
                new_texts[text] = texts[text]
        return new_texts

    def get_texts(self, texts, start, end):
        new_texts = {}
        keys = list(texts.keys())[start:end]
        for key in keys:
            new_texts[key] = texts[key]
        return new_texts

    def remove_emails(self, text):
        re_email = r'\S*@\S*\s?'
        pattern = re.compile(re_email)
        text = pattern.sub('', text)
        return text

    def remove_stopwords(self, text):
        stopwords = set(nltk.corpus.stopwords.words('portuguese'))
        words = [word for word in text if word not in stopwords]
        return " ".join(words)

    def pos_annotation(self, text):
        tagger = pickle.load(open("tagger.pkl", "rb"))
        portuguese_sent_tokenizer = nltk.data.load("tokenizers/punkt/portuguese.pickle")
        sentences = portuguese_sent_tokenizer.tokenize(text)
        tags = [tagger.tag(nltk.word_tokenize(sentence)) for sentence in sentences]
        # tags = [tagger.tag(sentence.split()) for sentence in sentences]
        return tags

    def lemmatize(self, text):
        cogroo = Cogroo.Instance()
        words = []
        texto = nltk.wordpunct_tokenize(text)
        for t in texto:
            t_aux = cogroo.lemmatize(t)
            words.append(t_aux)
        return " ".join(words)

    def create_bag_of_words(self, textos):
        unigrams = dict()
        bigrams = dict()
        trigrams = dict()
        stopwords = set(nltk.corpus.stopwords.words('portuguese'))
        for texto in textos:
            for i in textos[texto]['ngrams']['unigram']:
                term = " ".join(i)
                term = term.lower()
                if term not in stopwords:
                    if term in unigrams:
                        unigrams[term] += 1
                    else:
                        unigrams[term] = 1
            for i in textos[texto]['ngrams']['bigrams']:
                term = " ".join(i)
                term = term.lower()
                if term not in stopwords:
                    if term in bigrams:
                        bigrams[term] += 1
                    else:
                        bigrams[term] = 1

            for i in textos[texto]['ngrams']['trigrams']:
                term = " ".join(i)
                term = term.lower()
                if term not in stopwords:
                    if term in trigrams:
                        trigrams[term] += 1
                    else:
                        trigrams[term] = 1

        return unigrams, bigrams, trigrams


if __name__ == "__main__":
    clean_files = EstruturarTextos()
