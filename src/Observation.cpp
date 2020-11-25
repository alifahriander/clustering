#include <random>
#include <chrono>
#include "Observation.h"


using namespace std;


Observation::Observation(VectorXd mean_vector, VectorXd variance_vector, unsigned int numberSamples, unsigned seed){
    x = mean_vector;
    y = VectorXd(numberSamples);
    assignments = VectorXd(numberSamples);

    variances = variance_vector;

    generator_observation.seed(seed);
    bernoulli_distribution dist(0.1);

    Observation::computeObservation(numberSamples);
    // Observation::computeObservation(numberSamples, dist);

    // cout << "Mean Vector:" << endl << x << endl;
    // cout << "Observation Vector: "<< endl << y << endl;  
    // cout << "Assignment Vector: " << endl << assignments << endl;
}

/*
* Draw sample from normal distribution
* @param mean
* @param variance
* @return sample 
*/
double Observation::normalDistribution(double mean, double variance){
    normal_distribution<double> distribution(mean, variance);
    return distribution(generator_observation);
}

/*
* Draw sample from normal distribution
* @param mean
* @param variance
* @return sample 
*/
unsigned int Observation::uniformDistribution(unsigned int min, unsigned int max){
    uniform_int_distribution<unsigned int> distrib(min, max); 
    return distrib(generator_observation);

}
/*
* Draw sample from normal distribution
* @param mean
* @param variance
* @return sample 
*/
// unsigned int Observation::bernoulliDistribution(float p){
//     bernoulli_distribution<unsigned int> distrib(min, max); 
//     return distrib(generator_observation);

// }

void Observation::computeObservation(unsigned int numberSamples){    
    for(unsigned int i=0; i<numberSamples; ++i){
        assignments(i) = uniformDistribution(0,x.rows()-1);
        y(i) = normalDistribution(x(assignments(i)), variances(assignments(i)));
    }
}
void Observation::computeObservation(unsigned int numberSamples, bernoulli_distribution dist){    
    for(unsigned int i=0; i<numberSamples; ++i){
        assignments(i) =  dist(generator_observation);
        y(i) = normalDistribution(x(assignments(i)), variances(assignments(i)));
    }
}

