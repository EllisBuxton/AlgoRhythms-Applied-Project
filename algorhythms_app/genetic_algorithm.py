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