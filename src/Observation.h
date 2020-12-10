#ifndef OBSERVATION_H
#define OBSERVATION_H
#include <iostream>
#include <eigen3/Eigen/Core>
#include <string>


using namespace Eigen;
using namespace std;

class Observation{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VectorXd x;

        VectorXd assignments;
        VectorXd y;

        VectorXd variances;

        default_random_engine generator_observation;

        Observation(VectorXd mean_vector, VectorXd variance_vector, unsigned int numberSamples, unsigned seed);
        Observation();

        double normalDistribution(double mean, double variance);
        unsigned int uniformDistribution(unsigned int min, unsigned int max);

        void computeObservation(unsigned int numberSamples);

};
#endif 