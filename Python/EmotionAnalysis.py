# This is a program for analysing the sentiment of an input sentence
# Author: Sophist Wu
# 15:40 2011-8-10
import time
before = time.strftime('%X %x')
print before
print "importing dependencies from nltk..."
import collections
from datetime import datetime
from nltk import metrics    # for defining function precision_recall(.., ..)
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import movie_reviews
from nltk.corpus import reuters
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy   # for later testing the accuracy of the classifier
from nltk.tokenize import word_tokenize   # for word_tokenize method
from nltk.probability import FreqDist, ConditionalFreqDist


def high_information_words(labelled_words, score_fn=BigramAssocMeasures.chi_sq, min_score=5):
	word_fd = FreqDist()
	label_word_fd = ConditionalFreqDist()
	for label, words in labelled_words:
		for word in words:
			word_fd.inc(word)
			label_word_fd[label].inc(word)
		n_xx = label_word_fd.N()
		high_info_words = set()
		for label in label_word_fd.conditions():
			n_xi = label_word_fd[label].N()
			word_scores = collections.defaultdict(int)
		for word, n_ii in label_word_fd[label].iteritems():
			n_ix = word_fd[word]
			score = score_fn(n_ii, (n_ix, n_xi), n_xx)
			word_scores[word] = score
		bestwords = [word for word, score in word_scores.iteritems() if score >= min_score]
		high_info_words |= set(bestwords)
	return high_info_words


# precision_recall function is one which calculates the precision and recall of the clssifier
def precision_recall(classifier, testfeats):
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)

        for i, (feats, label) in enumerate(testfeats):
                refsets[label].add(i)
        precisions = {}
        recalls = {}

        for label in classifier.labels():
                precisions[label] = metrics.precision(refsets[label], testsets[label])
                recalls[label] = metrics.recall(refsets[label], testsets[label])
        return precisions, recalls

def bag_of_words(words):
	return dict([(word, True) for word in words])


def bag_of_words_not_in_set(words, badwords):
	return bag_of_words(set(words) - set(badwords))

def bag_of_non_stopwords(words, stopfile='english'):
	badwords = stopwords.words(stopfile)
	return bag_of_words_not_in_set(words, badwords)

def bag_of_bigrams_words(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
	bigram_finder = BigramCollocationFinder.from_words(words)
	bigrams = bigram_finder.nbest(score_fn, n)
	return bag_of_words(words + bigrams)

def label_feats_from_corpus(corp, feature_detector=bag_of_words):
	label_feats = collections.defaultdict(list)
	for label in corp.categories():
		for fileid in corp.fileids(categories=[label]):
			feats = feature_detector(corp.words(fileids=[fileid]))
			label_feats[label].append(feats)
	return label_feats
	
def split_label_feats(lfeats, split=0.75):
	train_feats = []
	test_feats = []
	for label, feats in lfeats.iteritems():
		cutoff = int(len(feats) * split)
		train_feats.extend([(feat, label) for feat in feats[:cutoff]])
		test_feats.extend([(feat, label) for feat in feats[cutoff:]])
	return train_feats, test_feats


def reuters_high_info_words(score_fn=BigramAssocMeasures.chi_sq):
	labeled_words = []
	for label in reuters.categories():
		labeled_words.append((label, reuters.words(categories=[label])))
	return high_information_words(labeled_words, score_fn=score_fn)

def reuters_train_test_feats(feature_detector=bag_of_words):
	train_feats = []
	test_feats = []
	for fileid in reuters.fileids():
		if fileid.startswith('training'):
			featlist = train_feats
		else:   # fileid.startswith('test')
			featlist = test_feats
		feats = feature_detector(reuters.words(fileid))
		labels = reuters.categories(fileid)
		featlist.append((feats, labels))
	return train_feats, test_feats

lfeats = label_feats_from_corpus(movie_reviews)

train_feats, test_feats = split_label_feats(lfeats)

# Define the classifier to analyse the emotion of the input sentence
nb_classifier = NaiveBayesClassifier.train(train_feats)




# Some tests
def a():
        while True:
                
               
                print "\nplease enter a sentense to be analysed:"
                sentence = raw_input('--> ')
                start = datetime.now().microsecond
                print "starting to analyse..."

                sent = word_tokenize(sentence)
                posfeat = bag_of_words(sent)

                result = nb_classifier.classify(posfeat)
                after = datetime.now().microsecond
                print "analyse finished..."
                print "result:", result
                timeconsuming = after-start
                print "It takes ", timeconsuming, "milliseconds to analyse the sentence"
                #print accuracy(nb_classifier, test_feats)

                #probs = nb_classifier.prob_classify(test_feats[0][0])
                #print probs.samples()
a()

                
#negfeat = bag_of_words(['baidu', 'sucks'])
#print nb_classifier.classify(negfeat)

# precision functioning
# nb_precisions, nb_recalls = precision_recall(nb_classifier, posfeat)
#accuracy(nb_classifier, posfeat)
#probs = nb_classifier.prob_classify(posfeat[0][0])
#print accuracy(nb_classifier, test_feats)

#nb_classifier.show_most_informative_features(n=100)

#from nltk.probability import LaplaceProbDist
#nb_classifier = NaiveBayesClassifier.train(train_feats, estimator=LaplaceProbDist)


#from nltk.probability import FreqDist, MLEProbDist, entropy
#fd = FreqDist({'pos': 30, 'neg': 10})
#print entropy(MLEProbDist(fd))

#fd['neg'] = 25
#print entropy(MLEProbDist(fd))

#fd['neg'] = 30
#print entropy(MLEProbDist(fd))

#fd['neg'] = 1
#print entropy(MLEProbDist(fd))



