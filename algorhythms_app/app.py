from flask import Flask, jsonify, render_template, request
from genetic_algorithm import initialize_population, evolve_population

app = Flask(__name__)

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


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)