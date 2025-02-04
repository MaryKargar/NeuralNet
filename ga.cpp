#include "nn.h"
#include <ga/GA1DArrayGenome.h>
#include <ga/GASimpleGA.h>
#include <ga/std_stream.h>
#include <ctime>
#include <cstdlib>
#include <iostream>


using namespace std;

const int GENOME_SIZE = 3;  // Number of integers in the genome
const int POPULATION_SIZE = 20;
const int MAX_GENERATIONS = 5;

// Objective function
float objective(GAGenome& g) {
    GA1DArrayGenome<int>& genome = (GA1DArrayGenome<int>&)g;
    double fitness = 1.0 / runRegression(genome.gene(0), genome.gene(1), genome.gene(2));
    return static_cast<float>(fitness);
}

// Initializer
void initializer(GAGenome& g) {
    GA1DArrayGenome<int>& genome = (GA1DArrayGenome<int>&)g;

    for (int i = 0; i < GENOME_SIZE; i++) {
        genome.gene(i, (rand() % 10 + 1) * 16);  // Initialize genes with random multiples of 16 between 16 and 160
    }
}

// Mutator
int mutator(GAGenome& g, float p) {
    GA1DArrayGenome<int>& genome = (GA1DArrayGenome<int>&)g;

    int nMutations = 0;
    for (int i = 0; i < GENOME_SIZE; i++) {
        if (GAFlipCoin(p)) {
            genome.gene(i, (rand() % 10 + 1) * 16);  // Mutate gene to a random multiple of 16 between 16 and 160
            nMutations++;
        }
    }

    return nMutations;
}

// Crossover
int crossover(const GAGenome& p1, const GAGenome& p2, GAGenome* c1, GAGenome* c2) {
    GA1DArrayGenome<int>& parent1 = (GA1DArrayGenome<int>&)p1;
    GA1DArrayGenome<int>& parent2 = (GA1DArrayGenome<int>&)p2;

    if (c1 && c2) {
        GA1DArrayGenome<int>& child1 = (GA1DArrayGenome<int>&)*c1;
        GA1DArrayGenome<int>& child2 = (GA1DArrayGenome<int>&)*c2;

        int cut = rand() % GENOME_SIZE;

        for (int i = 0; i < cut; i++) {
            child1.gene(i, parent1.gene(i));
            child2.gene(i, parent2.gene(i));
        }

        for (int i = cut; i < GENOME_SIZE; i++) {
            child1.gene(i, parent2.gene(i));
            child2.gene(i, parent1.gene(i));
        }

        return 2;
    } else if (c1) {
        GA1DArrayGenome<int>& child = (GA1DArrayGenome<int>&)*c1;
        int cut = rand() % GENOME_SIZE;

        for (int i = 0; i < cut; i++) {
            child.gene(i, parent1.gene(i));
        }

        for (int i = cut; i < GENOME_SIZE; i++) {
            child.gene(i, parent2.gene(i));
        }

        return 1;
    } else {
        return 0;
    }
}

int main() {
    srand(static_cast<unsigned int>(time(nullptr)));

    GA1DArrayGenome<int> genome(GENOME_SIZE, objective);
    genome.initializer(initializer);
    genome.mutator(mutator);
    genome.crossover(crossover);

    GASimpleGA ga(genome);
    ga.populationSize(POPULATION_SIZE);
    ga.nGenerations(MAX_GENERATIONS);
    ga.pMutation(0.2);
    ga.pCrossover(0.1);
    ga.evolve();

    const GA1DArrayGenome<int>& bestGenome = (GA1DArrayGenome<int>&)ga.statistics().bestIndividual();
    cout << "Best solution found: "<< endl;
    cout << "H1: "<< bestGenome.gene(0) << endl;
    cout << "H2: "<< bestGenome.gene(1)  << endl;
    cout << "H3: "<< bestGenome.gene(2)  << endl;

    double mse = runRegression(bestGenome.gene(0), bestGenome.gene(1), bestGenome.gene(2));
    cout << "Mean Squared Error on Prediction data points: " << mse << endl;

    return 0;
}
