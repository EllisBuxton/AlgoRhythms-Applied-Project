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
