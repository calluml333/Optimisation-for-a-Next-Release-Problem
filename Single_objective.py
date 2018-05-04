import matplotlib.pyplot as plt
from operator import itemgetter
import random
from timeit import default_timer


#===============================================================================
#=============================== Parameters ====================================

classic = 'Location of classic dataset'
realistic = 'Location of realistic dataset'


def read_data(File):
    """ 
    Reads in the csv file.
    """
    
    with open(File, 'r') as data:
        all_lines = [[int(num) for num in line.split()] for line in data]       
    return all_lines



data1 = read_data(realistic)
data2 = read_data(classic)


# Both of the splits of the dat abelow were hard-coded for each specific dataset
requirements_cost_realistic = data1[2]
data_realistic = data1[5:]

requirements_cost_classic = requirements_cost_classic = data2[2] + data2[4] + data2[6]
data_classic = customers_requirements_classic = data2[569:]




#===============================================================================
#=============================== Paratmeters ===================================

pop_size = 1000
generations = 200
select_size = 0.5*pop_size
weight = 0.6
cross_rate = 0.75
mutation_rate, mutation_prob = 0.05, 0.1

#data = data_classic
#requirements_cost = requirements_cost_classic

data = data_realistic
requirements_cost = requirements_cost_realistic



#===============================================================================
#============================ Sorting Functions ================================


def customer_info(data):
    """
    Scores the customers requirements and calculates the weight of each 
    customer s.t. the sum of the weights equals 1. 
    """
    
    customer_specs = []
    customer_profit = []
    for customer in data:
        profit = customer[0]
        customer_profit.append(profit)
    total = sum(customer_profit)
    weights = []
    for i in data:
        customer_weight = float(i[0])/float(total)
        weights.append(customer_weight)
        requirements = i[2:]
        value_array = []
        j=0
        while j < len(requirements):
            index = j + 1 
            value = 1./float(index)
            requirement = requirements[j]
            value_array.append([requirement,value])
            j += 1
        customer_specs.append([customer_weight,value_array])
    return customer_specs          


def requirement_score(customer_info, n_requirements):
    """
    Calculates the score of each requirement.
    """
    
    i = 0
    scores = []
    while i < n_requirements:
        score_list = []
        requirement_number = i + 1
        
        for customer in customer_info:
            weight = customer[0]
            reqs_list = customer[1]
            for reqs in reqs_list:
                if reqs[0] == requirement_number:
                    score_for_customer = weight*reqs[1]
                    score_list.append(score_for_customer)
        requirement_score = sum(score_list)
        scores.append((requirement_number, requirement_score))
        i += 1
    return scores
   
     
#===============================================================================             
#================================ Population =================================== 


def x_vector(n_requirements):
    x_vector = []
    while len(x_vector) < n_requirements:
        choice = random.randint(0,1)
        x_vector.append(choice)
    return x_vector
        

def initial_pop(pop_size, n_requirements):
    """
    Generates an initial population (of size "pop_size") of x vectors for the n
    requirements. 1 indicates that requirement is selected, 0 indicates it was
    not.
    """
    
    j = 0
    population = []
    while j < pop_size:
        newTest = x_vector(n_requirements)
        population.append(newTest)
        j += 1
    return population
    
    
#===============================================================================
#================================= Fitness =====================================


def fitness_score(xvec, reqs_scores):
    
    scores = []
    i = 0
    while i < len(xvec):
        if xvec[i] == 1:
            scores.append(reqs_scores[i][1])
        i += 1        
    xvec_fitness = sum(scores)
    return xvec_fitness 
    

def fitness_cost(xvec, reqs_costs):
    
    costs = []
    i = 0
    while i <len(xvec):
        if xvec[i] == 1:
            costs.append(reqs_costs[i])
        i += 1
    xvec_fitness = 1./float(sum(costs))  # low cost <=> high fitness
    return xvec_fitness


def fitness_function(population, reqs_score, reqs_cost, fit_weight):
    
    fit = []
    for x in population:
        fit_score = fitness_score(x, reqs_score)
        fit_cost = fitness_cost(x, reqs_cost)
        if fit_score == 0 or fit_cost == 0 or fit_score == 0 and fit_cost == 0:
            fitness =  0
        else:
            fitness =  (fit_score**fit_weight) * (fit_cost**(1-fit_weight))
                    
                                    
        fit.append([x, fitness, (fit_score, fit_cost)])
    fit_sorted = sorted(fit, key=itemgetter(1), reverse=True) #Sorts based on fitness
    return fit_sorted 
    


#===============================================================================
#================================ Tournament ===================================


def tournament(fitness, select_size):
    """
    First chooses the fittest 10 percent of the input population. The remaining
    90 percent are then put through a tournament selection process to create a 
    new population for crossover. The tournament will finish when the new 
    population size reaches 0.4*(length of input population).
    
    The new population is then combined with the fittest 10 percent to create 
    the Selected population, which is half the size of the input population.  
    """
    
    elite = int(0.1*len(fitness))
    select = fitness[:elite]
    winners = []
    k = 0
    while k <= select_size - len(select):
        i = random.randrange(elite, len(fitness))
        j = random.randrange(elite, len(fitness))
        if fitness[i][1] > fitness[j][1]:
            winners.append(fitness[j])
        else:
            winners.append(fitness[i])
        k += 1
    select = [select[y][0] for y in range(len(select))]
    winners = [fitness[x][0] for x in range(len(fitness))]
    select.extend(winners)
    return select

                    
#===============================================================================
#================================= Crossover ===================================

    
def crossover(pop, pop_size, cross_rate):
    """
    Two test suites are randomly selected from pop as parents. A random number 
    between 0 and 1 is then generated, and if this random number is less than 
    "Crossover_rate", the two parents are crossed over according to the methoud 
    outlined in Lecure 2. The two parents and the two children are then added
    to the "kids" list.
    
    If the random number generated is less that "Crossover_rate", the two
    parents chosen do not cross over and are added to the "kids" list.
    
    The original "pop" list is then extended to include the "kids " list.   
    """
    new_gen = []
    while len(new_gen) < (pop_size):
        parent_1 = random.choice(pop)
        parent_2 = random.choice(pop)        
        if random.random() < cross_rate:
            x = random.randint(1, len(parent_1) - 2)        
            child_1 = parent_1[:x] + parent_2[x:]
            child_2 = parent_2[:x] + parent_1[x:]
            new_gen.append(child_1)
            new_gen.append(child_2)            
        else:
            new_gen.append(parent_1)
            new_gen.append(parent_2)          
    if len(new_gen) > pop_size:
        new_gen = new_gen[:pop_size]
    return new_gen   
    
    
      
#===============================================================================
#================================ Mutation =====================================


def mutation(pop, rate, prob):
    """
    According to the "rate" of mutaiton, test suites are selected for mutation.
    For each of the test suites, a test from each is swapped with a randomly
    selected test from the "pool" of possible tests. 
    """    
    random.shuffle(pop)
    x = int(len(pop)*rate)
    pop_to_mutate = pop[:x]
    new_pop = pop[x:]
    i = 0
    while i < len(pop_to_mutate):
        j = 0
        while j < len(pop_to_mutate[i]):
            if random.random() < prob:
                if pop_to_mutate[i][j] == 0:
                    mutation = 1
                else:
                    mutation = 0
                pop_to_mutate[i] = pop_to_mutate[i][0:j] + [mutation] + pop_to_mutate[i][j + 1: len(pop_to_mutate[i])]
            j += 1
        i += 1
    new_pop_final = new_pop + pop_to_mutate
    random.shuffle(new_pop_final)
    return new_pop_final


#===============================================================================
#=============================== Pareto Front ==================================


def pareto_front(fitness):
    pareto_front = []
    other_pop = []
    pareto_front.append(fitness[0]) # adds the first fitness to PF (as will have highest score)
    i = 0
    j = 1
    while len(fitness) - (i+j) != 0:
        test_case = fitness[i]        
        if fitness[i+j][2][1] > test_case[2][1] or fitness[i+j][2] == test_case[2]:            
            pareto_front.append(fitness[i+j])
            i = i + j  #now test case will become the newly added solution
            j = 1      #to compare the new test case with the next in the list                     
        else:
            other_pop.append(fitness[i+j])
            j += 1     #move to next solution and keep comparing 
    return pareto_front
    
 
#===============================================================================   
#================================ Algorithm ====================================


def genetic_algorithm(data, reqs_cost, pop_size, generations , fit_weight, select_size, cross_rate, mutation_rate, mutation_prob):
        start = default_timer()
        cust_info = customer_info(data)
        n_requirements = len(reqs_cost)
        reqs_score = requirement_score(cust_info, n_requirements)
        population = initial_pop(pop_size, n_requirements)
        score_fit_initial = []
        cost_fit_initial = []
        score_fit = []
        cost_fit = []
        score_fit_final = []
        cost_fit_final = []
        fitness_history = []
        
        i = 1
        while i <= generations:
            print("Generation", i)
            fitness = fitness_function(population, reqs_score, reqs_cost, fit_weight)
            fitness_history.extend(fitness)         
                     
            winners = tournament(fitness, select_size)
            new_gen = crossover(winners, pop_size, cross_rate)
            population = mutation(new_gen, mutation_rate, mutation_prob)
            if i == 1:
                for fit in fitness:
                    score_fit_initial.append(fit[2][0])
                    cost_fit_initial.append(1./float(fit[2][1]))
            elif i == generations:
                for fit in fitness:
                    score_fit_final.append(fit[2][0])
                    cost_fit_final.append(1./float(fit[2][1]))
            else:
                for fit in fitness:
                    score_fit.append(fit[2][0])
                    cost_fit.append(1./float(fit[2][1]))
            i += 1       
          
        print("\nLength of final pop", len(population), "|| Time Taken:", "{}s".format(default_timer() - start), "|| Fitness weight =", fit_weight)

        #plt.plot(cost_fit, score_fit, 'k.')#, label = 'Final Pop w = "{}"'.format(fit_weight))
        plt.plot(cost_fit_initial, score_fit_initial, 'b.') # plots initial (random) population
        #plt.plot(cost_fit_final, score_fit_final, 'r+') 
        

        pf_fitness = sorted(fitness_history, key=itemgetter(2), reverse=True)  #sorts fintess baed on score for input into PF function
        pareto_supremo = pareto_front(pf_fitness)
        print("Length of Pareto Supremo", len(pareto_supremo))
        pf_score = []
        pf_cost = []
        for solution in pareto_supremo:
            pf_score.append(solution[2][0])
            pf_cost.append(1./float(solution[2][1]))
        plt.plot(pf_cost, pf_score, 'gx-', label = "Final S.O. Pareto Front")
        plt.title("Final Parteo Front for S.O. Algorithms")
        plt.legend()    
        plt.show()
        
        return population



genetic_algorithm(data, requirements_cost, pop_size, generations, weight, select_size, cross_rate, mutation_rate, mutation_prob)





