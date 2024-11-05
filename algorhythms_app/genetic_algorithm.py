import random

# Function to generate a random melody
def generate_random_melody(length=8):
    return [random.randint(60, 72) for _ in range(length)]

# Function to initialize a population of random melodies
def initialize_population(population_size=10, melody_length=8):
    return [generate_random_melody(melody_length) for _ in range(population_size)]

# Function to calculate the fitness of a melody
def calculate_fitness(melody_index, ratings_dict):
    ratings = ratings_dict.get(melody_index, [])
    if not ratings:
        return 0  # Default fitness if no ratings yet
    return sum(ratings) / len(ratings)  # Average rating as fitness score

# Selection: Select melodies based on their fitness scores
def selection(population, fitness_scores, num_to_select=5):
    # Select top num_to_select melodies based on fitness
    selected = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
    return [melody for melody, score in selected[:num_to_select]]

# Crossover: Combine two parent melodies to create a child melody
def crossover(parent1, parent2):
    # Simple one-point crossover
    crossover_point = len(parent1) // 2
    return parent1[:crossover_point] + parent2[crossover_point:]

# Mutation: Randomly mutate a melody
def mutate(melody, mutation_rate=0.1):
    for i in range(len(melody)):
        if random.random() < mutation_rate:
            melody[i] = random.randint(60, 72)  # Mutate the note
    return melody

# Function to evolve the population
def evolve_population(population, ratings_dict, num_generations=10, mutation_rate=0.1):
    for generation in range(num_generations):
        # Step 1: Calculate fitness scores based on actual ratings
        fitness_scores = [calculate_fitness(i, ratings_dict) for i in range(len(population))]

        # Step 2: Select the top melodies
        selected = selection(population, fitness_scores)

        # Step 3: Create next generation through crossover and mutation
        next_generation = []
        while len(next_generation) < len(population):
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            next_generation.append(child)

        population = next_generation
    return population