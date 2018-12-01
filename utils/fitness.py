import numpy as np
from itertools import product

def calcMatrix(lTweets):
	#print(lTweets)
	tam = len(lTweets)
	matriz = np.zeros((tam, tam))
	for i in range(len(lTweets)):
		for j in range(i, len(lTweets)):
			matriz[i][j] = jaccard(lTweets[i],lTweets[j])
	return matriz

def jaccard(string1, string2):
	set1 = set(string1.split())
	set2 = set(string2.split())
	return 1-(len( set1.intersection(set2) ) / len( set1.union(set2) ))
