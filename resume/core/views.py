from django.shortcuts import render, redirect
from django.views.generic import TemplateView, ListView, CreateView
from django.core.files.storage import FileSystemStorage
from django.urls import reverse_lazy
import json
import glob
import os
import re
import pytesseract
from pyresparser import ResumeParser
from pytesseract import image_to_string
import textract as tx
from nltk import word_tokenize

from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import math
import numpy as np

"""std_embeddings_index = {}
with open('resume/core/numberbatch-en.txt') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        std_embeddings_index[word] = embedding"""


class Home(TemplateView):
    template_name = 'home.html'


def upload(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        #regex = re.compile(r'[\n\r\t\\u2013]')
        name = fs.save(uploaded_file.name, uploaded_file)
        context['url'] = fs.url(name)
        latest_resume = newest('media')
        jd = ('resume/core/Java Developer -Job Description .pdf')
        jd_text = ResumeParser(jd).get_extracted_data()
        resume_text = ResumeParser(
            latest_resume, skills_file='media/skills.csv').get_extracted_data()
        score = findsimilarity(
            listtostr(resume_text['skills']), listtostr(jd_text['skills']))
        #score=cosine_sim(s1, s2)
        context['resume_text'] = resume_text
        context['jd_text'] = jd_text
        context['score'] = score
    return render(request, 'upload.html', context)


def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)


def file_to_string(filepath):
    text = tx.process(filepath)
    text = text.decode('utf-8')
    return text


def findsimilarity(X, Y):
    X_list = word_tokenize(X)
    Y_list = word_tokenize(Y)

# sw contains the list of stopwords
    l1 = []
    l2 = []

# remove stop words from the string
    X_set = set(X_list)
    Y_set = set(Y_list)

# form a set containing keywords of both strings
    rvector = X_set.union(Y_set)
    for w in rvector:
        if w in X_set:
            l1.append(1)  # create a vector
        else:
            l1.append(0)
        if w in Y_set:
            l2.append(1)
        else:
            l2.append(0)
    c = 0

# cosine formula
    for i in range(len(rvector)):
        c += l1[i]*l2[i]
    cosine = c / float((sum(l1)*sum(l2))**0.5)

    return cosine


"""def preprocess(words):
    words = normalize(words)
    words = ' '.join(map(str, words))
    return words"""


def get_word2vec(resume, jd):
    word_emb_model = Word2Vec.load('word2vec.bin')
    return word_emb_model.similarity(resume, jd)


def listtostr(s):
    if not s:
        return " "
    listToStr = ' '.join(map(str, s))
    return listToStr


def get_cosine_similarity(feature_vec_1, feature_vec_2):
    return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]


"""def cosine_sim(sent1, sent2):
    return cosineValue(get_sentence_vector(sent1), get_sentence_vector(sent2))


def cosineValue(v1, v2):

    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)


def get_sentence_vector(sentence, std_embeddings_index=std_embeddings_index):
    sent_vector = 0
    for word in sentence.lower().split():
        if word not in std_embeddings_index:
            word_vector = np.array(np.random.uniform(-1.0, 1.0, 300))
            std_embeddings_index[word] = word_vector
        else:
            word_vector = std_embeddings_index[word]
        sent_vector = sent_vector + word_vector

    return sent_vector"""
