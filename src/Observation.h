#ifndef OBSERVATION_H
#define OBSERVATION_H
#include <iostream>
#include <eigen3/Eigen/Core>


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
        double normalDistribution(double mean, double variance);
        void computeObservation(unsigned int numberSamples);
        unsigned int uniformDistribution(unsigned int min, unsigned int max);

};
#endif 