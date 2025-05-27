import numpy as np
import matplotlib.pyplot as plt

def fitness_function(x):
    return x ** 2

def create_population(pop_size, x_range):
    return np.random.uniform(low=x_range[0], high=x_range[1], size=pop_size)

def select_parents(population, fitness_values, num_parents):
    parents = population[np.argsort(fitness_values)[-num_parents:]]
    return parents

def crossover(parents, num_offspring):
    offspring = []
    for i in range(num_offspring):
        parent1, parent2 = parents[i % len(parents)], parents[(i + 1) % len(parents)]
        offspring.append([parent1, parent2])  # Keep parents as single values within lists
    return np.array(offspring).flatten()

def mutate(offspring, mutation_rate, x_range):
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            offspring[i] = np.random.uniform(low=x_range[0], high=x_range[1])
    return offspring

def genetic_algorithm(pop_size, num_generations, x_range, mutation_rate, num_parents):
    population = create_population(pop_size, x_range)
    fitness_history = []
    population_history = []
    
    for generation in range(num_generations):
        fitness_values = np.array([fitness_function(x) for x in population])
        fitness_history.append(np.max(fitness_values))
        population_history.append(population.copy())
        
        parents = select_parents(population, fitness_values, num_parents)
        offspring = crossover(parents, pop_size - num_parents)
        population = np.concatenate((parents, offspring))
        population = mutate(population, mutation_rate, x_range)
    
    return population, fitness_history, population_history

# Set algorithm parameters
pop_size = 50
num_generations = 100
x_range = (-10, 10)
mutation_rate = 0.1
num_parents = 10

# Run genetic algorithm
final_population, fitness_history, population_history = genetic_algorithm(
    pop_size, num_generations, x_range, mutation_rate, num_parents
)

# Create figure with two subplots
plt.figure(figsize=(12, 5))

# Plot 1: Fitness Progress
plt.subplot(1, 2, 1)
plt.plot(fitness_history)
plt.title('Genetic Algorithm Fitness Progress')
plt.xlabel('Generation')
plt.ylabel('Max Fitness')
plt.grid(True)

# Plot 2: Population Distribution
plt.subplot(1, 2, 2)
plt.hist(final_population, bins=20, edgecolor='black')
plt.title('Final Population Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)

# Adjust layout and display
plt.tight_layout()
plt.show()