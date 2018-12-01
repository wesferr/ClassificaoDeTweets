#!/usr/bin/env python3
import numpy as np
from Genetic import Genetic_algoritm
from Tweet import read_tweets
from utils.fitness import calcMatrix
from utils.wordcloud import createCloud

def main():
    
    tweets = read_tweets("Tweets.json")
    matrix = calcMatrix([i.text for i in tweets])
    
    n_clusters=[5, 10, 25]
    n_populacao=[100, 150, 300, 350]
    n_geracoes=[10000, 30000, 60000]
    
    for g in n_geracoes:
      for p in n_populacao:
        for c in n_clusters:
                print('\n\nAlgoritmo genético com Clusters: {} Populacao: {} Geracoes: {}'.format(c, p, g))

                genetic = Genetic_algoritm(tweets, c, matrix) # n_population, n_generations
                best = genetic.run(p, g)
                
                print("MELHOR  FITNEES", best.fitness)
                string = 'C {} - E {:.2f} - G {} - P {}'.format(c, best.fitness,g, p)
                gravar_txt(string+'.txt', tweets,  best)    
                gravar_wordcloud(best,tweets, '/'+string)
    
def gravar_wordcloud(best, tweets, intermediary_path):
    bestt = np.array(best.matrix_membership).T 
    separated_tweets= []
    for i, x in enumerate(bestt):
        separated_tweets.append([])
        for j, y in enumerate(x):
            if(bestt[i][j]==1):
                separated_tweets[i].append(tweets[j])
    createCloud(separated_tweets, intermediary_path)   

def gravar_txt(nomearquivo, tweets, best):
    matrix = []
    n_cluster=len(best.matrix_membership[0])
    
    for i in range(n_cluster):
        matrix.insert(i, [])
    
    for i in range(len(tweets)): 
        pos = np.amin(np.nonzero(best.matrix_membership[i]))
        matrix[pos].append(i)           
        
    arquivo = open("texto/"+nomearquivo, 'w')
    
    for i in range(n_cluster):
        
        tweets_cluster = []
        arquivo.write('\n\nCLUSTER {}'.format(i))
        
        for k in range(len(matrix[i])):
            pos = matrix[i][k]
            tweets_cluster.append(tweets[pos].text)
            
        tweets_cluster.sort() 
        
        for k in range(len(tweets_cluster)):
            arquivo.write('\n{}'.format(tweets_cluster[k]))
        
    arquivo.close()

if(__name__ == "__main__"):
    main()
