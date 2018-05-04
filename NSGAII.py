import matplotlib.pyplot as plt
from operator import itemgetter
import random
from timeit import default_timer


#===============================================================================
#============================= Data Formatting =================================

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


population_size = 1000
selection_size = int(0.5*population_size)
generations = 200
crossover_rate = 0.75
mutation_rate, mutation_prob = 0.05, 0.1


## Select the data you want to use

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


def fitness_one(xvec, reqs_scores):
    """
    Assesses the fitness of the xvectors score.
    """
    scores = []
    i = 0
    while i < len(xvec):
        if xvec[i] == 1:
            scores.append(reqs_scores[i][1])
        i += 1        
    xvec_fitness = sum(scores)
    return xvec_fitness 
    

def fitness_two(xvec, reqs_costs):
    """
    Assesses the fitness of the 
    """
    
    costs = []
    i = 0
    while i <len(xvec):
        if xvec[i] == 1:
            costs.append(reqs_costs[i])
        i += 1
    xvec_fitness = 1./float(sum(costs))  # low cost <=> high fitness
    return xvec_fitness


def fitness_function(population, reqs_score, reqs_cost):
    
    fit = []
    for x in population:
        fit_score = fitness_one(x, reqs_score)
        fit_cost = fitness_two(x, reqs_cost)
        fit.append([x, (fit_score, fit_cost)])
    fit_sorted = sorted(fit, key=itemgetter(1), reverse=True) #Sorts based on fit_score
    return fit_sorted 
    



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
        if fitness[i+j][1][1] > test_case[1][1] or fitness[i+j][1] == test_case[1]:            
            pareto_front.append(fitness[i+j])
            i = i + j  #now test case will become the newly added solution
            j = 1      #to compare the new test case with the next in the list                     
        else:
            other_pop.append(fitness[i+j])
            j += 1     #move to next solution and keep comparing 
    return (pareto_front, other_pop)
                   

def fronts(fitness):
    front_array = []
    others = fitness[:]
    while len(others) != 0:
        pf = pareto_front(others)       
        front_array.append(pf[0])
        others = pf[1]   
    return front_array




#===============================================================================
#================================ Selection ====================================


def selection_series(series_length, length_of_front):
    """
    Creates the series [0, 1, 1/2, 1/4, 3/4, 1/8, 7/8, 3/8, 5/8, ...], making it
    the same length as some specified input length. 
    """
    
    series = [0, length_of_front - 1, int(0.5*length_of_front)]
    i = 1 #start from i = 1 as we already included 0.5 (i = 0) in the initial series
    while len(series) < series_length:
        j = 0
        while j < (2**i / 2):
            fraction_1 = int((2. * float(j) + 1.) / (2. ** (float(i) + 1.))*length_of_front)
            series.append(fraction_1) 

            fraction_2 = int((1. - (2. * float(j) + 1.) / (2. ** (float(i) + 1.)))*length_of_front)
            series.append(fraction_2)           
            j += 1     
        i+= 1   
    if len(series) > series_length:
        series = series[:series_length]   
    return series
            

def selection(fronts, select_size):
    """
    Starting form non-dominated, adds in each front to the new population until
    it reaches a front that contains more solutions than required.
    
    From this front
    """
    
    new_pop = []
    duplicate_indexes = []
    i = 0            
    sum_length_of_fronts = 0  
    while i < len(fronts):
        front = fronts[i]
        sum_length_of_fronts += len(front)
        if sum_length_of_fronts < select_size:    
            for vector in front:
                new_pop.append(vector[0])   #adds in all elements of the front to new pop
            i += 1     
        elif sum_length_of_fronts > select_size:
            final_front = front[:]            # creates a copy for use in selection
            duplicate_final_front = front[:]  # creates a copy to monitor what elements from front are not already included in new pop
            number_to_be_added = select_size - len(new_pop)   
            solutions_to_select = selection_series(number_to_be_added, len(final_front))  # generates our sequence of selection values
            for value in solutions_to_select:   
                solution_about_to_add = final_front[value]    
                if solution_about_to_add not in new_pop:      #if soution is not already in new pop
                    new_pop.append(solution_about_to_add[0])  #add it 
                    if solution_about_to_add in duplicate_final_front:
                        duplicate_final_front.remove(solution_about_to_add) #remove it form the duplicate front
                else:                                         #if soultion already in new pop
                    duplicate_indexes.extend(value)           # choose a random element from duplicate and add
                    new_pop.append(random.choice(duplicate_final_front))                 
            i = len(fronts)     
    if len(new_pop) > select_size:    #to double check new pop is not too long.
        new_pop = new_pop[:select_size]    
    return new_pop


                    
                                                            
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
#================================ Algorithm ====================================


def genetic_algorithm(data, reqs_cost, pop_size, generations, select_size, cross_rate, mutation_rate, mutation_prob):
    start = default_timer()

    PF_History = []
    score_fit = []
    cost_fit = []
       
    cust_info = customer_info(data)     #output [[customer weight, [[req_no,importance]]]]
    n_requirements = len(reqs_cost)
    reqs_score = requirement_score(cust_info, n_requirements)    #output [(req_no. score)]
        
    population = initial_pop(pop_size, n_requirements)    #output [xvecs]
    fitness = fitness_function(population, reqs_score, reqs_cost)
    
    score_fit_initial = []
    cost_fit_initial = []
    for m in fitness:
        score_fit_initial.append(m[1][0])
        cost_fit_initial.append(1./float(m[1][1]))
    
    pf_fronts = fronts(fitness) 
    PF_History.extend(pf_fronts[0])       #add in the initial non dominated front
    
    winners = selection(pf_fronts, select_size)
    new_gen = crossover(winners, pop_size, cross_rate)
    population = mutation(new_gen, mutation_rate, mutation_prob)
    i = 1
    gen = 1
    
    while i < generations:
        print("Generation", str(gen))
        fitness = fitness_function(population, reqs_score, reqs_cost)
        for fit in fitness:
            score_fit.append(fit[1][0])
            cost_fit.append(1./float(fit[1][1]))

        pf_fronts = fronts(fitness)
        PF_History.extend(pf_fronts[0])       

        winners = selection(pf_fronts, select_size)
        new_gen = crossover(winners, pop_size, cross_rate)
        population = mutation(new_gen, mutation_rate, mutation_prob)

        gen += 1       
        i += 1
        
        
    ### Plotting Stuff ###
        

    PF_History = sorted(PF_History, key = itemgetter(1), reverse = True)
    pareto_supremo = fronts(PF_History)               
    pf_score = []
    pf_cost = []  
    for fit_vector in pareto_supremo[0]:   #creates pareto supremo coords
        pf_score.append(fit_vector[1][0])
        pf_cost.append(1./fit_vector[1][1])
        
    plt.plot(cost_fit_initial, score_fit_initial, 'k.', label = "Random Populaiton")
    #plt.plot(cost_fit, score_fit, 'k.', label = "Population")
    plt.plot(pf_cost, pf_score, 'rx-', label = "Final NSGAII Pareto Front")                                        
    plt.xlabel("Cost")
    plt.ylabel("Score")
    plt.title("Final Parteo Front for the NSGAII")
    plt.legend()
    plt.show() 
    print("Length of final population", len(population), "|| Time Taken:", "{}s".format(default_timer() - start))
    return population



genetic_algorithm(data, requirements_cost, population_size, generations, selection_size, crossover_rate, mutation_rate, mutation_prob)







