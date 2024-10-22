import random

# Function to generate a random melody
def generate_random_melody(length=8):
    return [random.randint(60, 72) for _ in range(length)]

# Function to initialize a population of random melodies
def initialize_population(population_size=10, melody_length=8):
    return [generate_random_melody(melody_length) for _ in range(population_size)]

# Function to simulate user ratings 
def mock_fitness_function(melody):
    # Placeholder for real fitness function based on user ratings
    return random.randint(1, 10)

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
def evolve_population(population, num_generations=10, mutation_rate=0.1):
    for generation in range(num_generations):
        # Calculate fitness scores
        fitness_scores = [mock_fitness_function(melody) for melody in population]

        # Select the top melodies
        selected = selection(population, fitness_scores)

        # Create next generation through crossover and mutation
        next_generation = []
        while len(next_generation) < len(population):
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            next_generation.append(child)

        population = next_generation
    return population