import os
import json

from glob import glob
from collections import OrderedDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

K = 50


class CriarWekaFiles:

    def __init__(self):
        directory_teste = os.path.join(BASE_DIR, 'CORPUS', 'TESTE')
        directory_treino = os.path.join(BASE_DIR, 'CORPUS', 'TREINO')
        teste_filenames = glob("{}/*.txt".format(directory_teste))
        treino_filenames = glob("{}/*.txt".format(directory_treino))

        directory_bow_treino = os.path.join(BASE_DIR, 'CORPUS', 'TREINO', 'BoW', 'All')
        directory_bow_teste = os.path.join(BASE_DIR, 'CORPUS', 'TESTE', 'BoW', 'All')
        # glob("{}/*.txt".format(directory_bow_treino))
        bow_treino_filenames = [os.path.join(directory_bow_treino, 'all_unigram.txt')]
        # glob("{}/*.txt".format(directory_bow_teste))
        bow_teste_filenames = [os.path.join(directory_bow_teste, 'all_unigram.txt')]

        self.textos_teste = {}
        self.textos_treino = {}

        self.bow_teste = OrderedDict()
        self.bow_treino = OrderedDict()

        self.ler_arquivos(teste_filenames, treino_filenames)
        self.ler_bow_files(bow_teste_filenames, bow_treino_filenames)
        self.criar_wekafiles_treino()
        self.criar_wekafiles_teste()

    def ler_bow_files(self, teste_filenames, treino_filenames):
        for fileteste in teste_filenames:
            with open(fileteste, 'r') as arq:
                lines = arq.readlines()
                for line in lines:
                    tmp_teste = line.split('-')
                    term = tmp_teste[0].strip().lstrip()
                    count = tmp_teste[1].strip().lstrip().replace('\n', '')
                    self.bow_teste[term] = count
        for filetreino in treino_filenames:
            with open(filetreino, 'r') as arq:
                lines = arq.readlines()
                for line in lines:
                    tmp_treino = line.split('-')
                    term = tmp_treino[0].strip().lstrip()
                    count = tmp_treino[1].strip().lstrip().replace('\n', '')
                    self.bow_treino[term] = count

    def ler_arquivos(self, teste_filenames, treino_filenames):
        for fileteste in teste_filenames:
            with open(fileteste, 'r') as arq:
                tmp_load = json.load(arq)
                self.textos_teste[fileteste.split('/')[-1]] = tmp_load
        for filetreino in treino_filenames:
            with open(filetreino, 'r') as arq:
                tmp_load = json.load(arq)
                self.textos_treino[filetreino.split('/')[-1]] = tmp_load

    def criar_header_wekafile(self, words, relation):
        header = list()
        header.append("@relation {}\n".format(relation))
        for i in words:
            header.append("@attribute {} integer\n".format(i))
        return header

    def criar_wekafiles_teste(self):
        words = list(self.bow_teste.keys())[0:K]
        lines = self.criar_header_wekafile(words, 'Arquivo')
        lines.append('@data\n')
        for arq in self.textos_teste:
            for texto in self.textos_teste[arq]:
                tmp_string = ""
                for word in words:
                    text = self.textos_teste[arq][texto]['texto']
                    if word in text:
                        tmp_string += "1, "
                    else:
                        tmp_string += "0, "
                tmp_string += "{}\n".format(arq.lower().replace(" ", "_"))
                lines.append(tmp_string)

        with open("Wekafiles/teste.arff", 'w') as arq:
            for line in lines:
                arq.write(line)

    def criar_wekafiles_treino(self):
        words = list(self.bow_treino.keys())[0:K]
        lines = self.criar_header_wekafile(words, 'Arquivo')
        lines.append('@data\n')
        for arq in self.textos_treino:
            for texto in self.textos_treino[arq]:
                tmp_string = ""
                for word in words:
                    text = self.textos_treino[arq][texto]['texto']
                    if word in text:
                        tmp_string += "1, "
                    else:
                        tmp_string += "0, "
                tmp_string += "{}\n".format(arq.lower().replace(" ", "_"))
                lines.append(tmp_string)

        with open("Wekafiles/treino.arff", 'w') as arq:
            for line in lines:
                arq.write(line)


if __name__ == "__main__":
    wekafiles = CriarWekaFiles()
