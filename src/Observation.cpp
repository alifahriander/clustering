#include <random>
#include <chrono>
#include "Observation.h"


using namespace std;


// Observation::Observation(VectorXd mean_vector, VectorXd variance_vector, unsigned int numberSamples, unsigned seed){
//     x = mean_vector;
//     y = VectorXd(numberSamples);
//     assignments = VectorXd(numberSamples);

//     variances = variance_vector;

//     generator_observation.seed(seed);

//     Observation::computeObservation(numberSamples);
// }

Observation::Observation(MatrixXd mean_matrix, MatrixXd variance_matrix, unsigned int numberSamples, unsigned seed){
    dimension = mean_matrix.cols();

    Y = MatrixXd(numberSamples,dimension);
    assignments = VectorXd(numberSamples);

    MatrixXd covariances[mean_matrix.rows()];

    for(unsigned int i=0; i<mean_matrix.rows(); ++i){

        Matrix<double,Dynamic,Dynamic,RowMajor> flatCovariance = MatrixXd(1,dimension*dimension);
        flatCovariance << variance_matrix.row(i);
        flatCovariance.resize(dimension,dimension);
        covariances[i] = flatCovariance;

    }


    MatrixXd choleskyMatrices[mean_matrix.rows()];
    for(unsigned int i=0; i<mean_matrix.rows(); ++i){
        LLT<MatrixXd> lltMatrix(covariances[i]);
        MatrixXd L = lltMatrix.matrixL();
        choleskyMatrices[i] = L;

    }

    generator_observation.seed(seed);

    Observation::computeObservation(mean_matrix, choleskyMatrices, numberSamples);
}


Observation::Observation(){

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

// void Observation::computeObservation(unsigned int numberSamples){    
//     for(unsigned int i=0; i<numberSamples; ++i){
//         assignments(i) = uniformDistribution(0,x.rows()-1);
//         y(i) = normalDistribution(x(assignments(i)), variances(assignments(i)));
//     }
// }

void Observation::computeObservation(MatrixXd meanMatrix, MatrixXd choleskyMatrices[], unsigned int numberSamples){
    MatrixXd observation(numberSamples,meanMatrix.cols());

    for(unsigned int i=0; i<numberSamples; i++){
        assignments(i) = (int) uniformDistribution(0,meanMatrix.rows()-1);
        VectorXd normalVector(dimension);
        // normalVector = N(0,I);
        for(unsigned int k=0; k<dimension;++k){
            normalVector(k) = normalDistribution(0.0,1.0);
        }
        //normalVector = L * normalVector;
        normalVector = choleskyMatrices[(unsigned int) assignments(i)] * normalVector;
        // m + normalVector
        observation.row(i) = (meanMatrix.row(assignments(i)).transpose() + normalVector).transpose();

    }
    Y = observation;
    Matrix<double,Dynamic,Dynamic,RowMajor> tmpY(Y);
    Map<RowVectorXd> flatY(tmpY.data(), tmpY.size());
    y = flatY.transpose();
}

