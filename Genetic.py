from random import randint, choice, uniform
import numpy as np
from operator import attrgetter
from Tweet import Tweet, read_tweets
from utils.fitness import jaccard
from copy import deepcopy

class Cromossome:

    def __init__(self, matrix_membership, dissimilarity_matrix, n_tweets, n_cluster):
        self.matrix_membership = matrix_membership
        self.dissimilarity_matrix = dissimilarity_matrix
        self.n_tweets = n_tweets
        self.n_cluster = n_cluster
        self.fitness = self.fitnesscalc()

    def fitnesscalc(self):
        erro = 0
        for i in range(self.n_cluster):
            P = 1e-6 + np.sum(self.matrix_membership[:,i]) / self.n_tweets
            somatorio = np.outer(self.matrix_membership[:,i], self.matrix_membership[:,i])
            somatorio = np.sum(somatorio * self.dissimilarity_matrix)
            erro += (1/( P * self.n_tweets)) * somatorio       
        return erro

    def updatefitness(self):
        self.fitness=self.fitnesscalc()

class Genetic_algoritm:

    def __init__(self, tweets,n_cluster, dissimilarity_matrix):
        self.tweets = tweets
        self.n_tweets = len(tweets)
        self.n_cluster=n_cluster
        self.population = []
        self.dissimilarity_matrix = dissimilarity_matrix
        self.prob_mutation=0.01 # entre 0 e 1
        
    def run(self, n_population, n_generations):
        self.population = self.create_population(n_population) # cria população aleatória
        sem_melhoria=0
        fitness=0
        for g in range(n_generations):          
            self.melhor = min(self.population, key=attrgetter('fitness'))
            pior_fitness = max(self.population, key=attrgetter('fitness')).fitness
            total_fitness = self.fitness_inverted_sum(pior_fitness) # soma o fitness da população para fazer a roleta apenas uma vez
            new_pop = []

            for i in range(0, n_population, 2):
                p1, p2 = self.parents_selection(total_fitness,pior_fitness)
                if p1!=p2:
                    child1, child2 = self.crossover(p1, p2)
                else:
                    child1, child2 = p1, p2
                    child1 = self.mutation(child1)                   
                if uniform(0, 1)<=self.prob_mutation: # sorteio para decidir se realiza mutação
                     child1 = self.mutation(child1)
                     child2 = self.mutation(child2)
                
                new_pop.append(child1)
                new_pop.append(child2)
                        
            self.population = new_pop
            self.elitism(self.melhor)
			
            if g%100 == 0:                      
                self.melhor = min(self.population, key=attrgetter('fitness'))
                print(g, self.melhor.fitness)
                      
                if fitness == self.melhor.fitness:
                    sem_melhoria+=1
                    if sem_melhoria==10:
                        self.extinction_colonization(int(n_population-(n_population/33)))
                        print("Ocoreu um desastre, só os 3% mais aptos sobreviveram!")
                    elif sem_melhoria==20:
                        print("Chegamos ao máximo!")
                        break
                else:
                    fitness = self.melhor.fitness
                    sem_melhoria=0
               
        return min(self.population, key=attrgetter('fitness'))

    # seleção de pais por roleta russa
    def parents_selection(self, fitness_sum, 	pior_fitness):
        pior = max(self.population, key=attrgetter('fitness'))      
        max_fitness =  pior.fitness + self.melhor.fitness
        p1 = self.roulette_selection(fitness_sum, max_fitness)
        p2 = self.roulette_selection(fitness_sum, max_fitness)
        return p1, p2

    def roulette_selection(self, fitness_sum, max_fitness):
        value = randint(0,int(fitness_sum))
        part_sum = 0
        for i in range(len(self.population)):
            part_sum += max_fitness-self.population[i].fitness
            if part_sum>=value:
                return self.population[i]

    def fitness_inverted_sum(self, pior_fitness):
        max_fitness =  pior_fitness+ self.melhor.fitness
        the_sum = 0

        for i in range(len(self.population)):            
            the_sum+=max_fitness-self.population[i].fitness # somando o fitness de toda população

        return the_sum

    # gera população de cromossomos com os clusters definidos aleatóriamente
    def create_population(self, n_population):
        population = []
        for i in range(n_population):
            #print("população: {}".format(i))
            population.append(self.rand_membership())
        return population

    # cria um Cromossomo contendo o membership aleatória de cada Tweet
    def rand_membership(self):
    	matrix_membership=np.zeros((self.n_tweets,self.n_cluster), dtype=np.byte)
    	for i in range(self.n_tweets):
    		matrix_membership[i][randint(0,self.n_cluster-1)] = 1
    	return Cromossome(matrix_membership, self.dissimilarity_matrix, self.n_tweets, self.n_cluster)

    # sorteia uma posição como ponto de quebra e realiza a cruza
    def crossover(self, p1, p2):
      pos = randint(1, int(self.n_tweets/2))
      pos2 = randint(pos+1, self.n_tweets-2)      
      dna1 = np.concatenate((np.array(p1.matrix_membership[:pos, :]), np.array(p2.matrix_membership[pos:pos2, :]),  np.array(p1.matrix_membership[pos2:, :])), axis=0)
      dna2 = np.concatenate((np.array(p2.matrix_membership[:pos, :]), np.array(p1.matrix_membership[pos:pos2, :]),  np.array(p2.matrix_membership[pos2:, :])), axis=0)
      return Cromossome(dna1, self.dissimilarity_matrix, self.n_tweets, self.n_cluster), Cromossome(dna2, self.dissimilarity_matrix, self.n_tweets, self.n_cluster)

    # para dar maior variabilidade ele só subestitui se o melhor não constar mais na população (caso não ocorreu crossover ele já está)
    def elitism(self, best): # substituir o melhor pelo pior 
        try:
            i = self.population.index(best)  
        except ValueError:
            worst =  max(self.population, key=attrgetter('fitness'))
            self.population.remove(worst)
            self.population.append(best)
    
    # extingue a parte menos apta da população e recoloniza
    def extinction_colonization(self, n_extints):          
        populacao = sorted(self.population, key = lambda c: c.fitness)
        self.population = populacao[:n_extints]
        for i in range(n_extints):
            self.population.append(self.rand_membership())
            
    def mutation(self, cromossome):
        for i in range (randint(1, self.n_cluster/5)):
            pos_tweet = randint(0, self.n_tweets-1)
            other_cluster = randint(0, self.n_cluster-1)
            cromossome.matrix_membership[pos_tweet:pos_tweet+1, :] = 0
            cromossome.matrix_membership[pos_tweet][other_cluster] = 1
        return Cromossome(cromossome.matrix_membership, self.dissimilarity_matrix, self.n_tweets, self.n_cluster)

  