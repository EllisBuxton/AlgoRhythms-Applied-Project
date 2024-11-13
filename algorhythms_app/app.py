from flask import Flask, jsonify, render_template, request
from genetic_algorithm import initialize_population, evolve_population
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Home route
@app.route('/')
def index():
    return render_template('index.html')


# Instruments route
@app.route('/instruments', methods=['GET', 'POST'])
def select_instruments():
    if request.method == 'POST':
        # Retreive Selected Instruments from the form
        selected_instruments = request.form.getlist('instruments')
        return jsonify({'instruments': selected_instruments})
    return render_template('instruments.html')

# Route to generate melodies
@app.route('/generate', methods=['GET'])
def generate_melodies():
    population_size = 10
    melody_length = 8
    population = initialize_population(population_size, melody_length)

    return jsonify({'melodies': population})

# In-memory storage for ratings
melody_ratings = {}

@app.route('/rate', methods=['POST'])
def rate_melody():
    data = request.json
    melody_index = data['melodyIndex']
    rating = int(data['rating'])

    # Store the rating for the melody in memory (or update if it already exists)
    if melody_index in melody_ratings:
        melody_ratings[melody_index].append(rating)
    else:
        melody_ratings[melody_index] = [rating]

    return jsonify({'message': f'Melody {melody_index + 1} rated {rating}'})

# Initialize population on app start
population_size = 10
melody_length = 8
population = initialize_population(population_size, melody_length)

# Route to evolve melodies based on ratings
@app.route('/evolve', methods=['GET'])
def evolve_melodies():
    global population
    population = evolve_population(population, melody_ratings, num_generations=5, mutation_rate=0.1)

    # Return the newly evolved population
    return jsonify({'melodies': population})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)