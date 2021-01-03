#include <random>
#include <chrono>
#include "Observation.h"


using namespace std;


Observation::Observation(){

}


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

    // Compute Cholesky Decomposition
    MatrixXd choleskyMatrices[mean_matrix.rows()];
    for(unsigned int i=0; i<mean_matrix.rows(); ++i){
        LLT<MatrixXd> lltMatrix(covariances[i]);
        MatrixXd L = lltMatrix.matrixL();
        choleskyMatrices[i] = L;

    }

    generator_observation.seed(seed);

    Observation::computeObservation(mean_matrix, choleskyMatrices, numberSamples);
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

/**
* Compute Observation for both 1-D and multidimensional case
* To compute the observation, we first choose from which cluster the observation is sampled by drawing a sample from a uniform distribution at random.
* (We decompose the corresponding variance matrix to LL^T with Cholesky Decomposition in the initialization function)
* A new sample is given by m + Lu, where m is a mean vector in the row vectors of @meanMatrix , L is the Cholesky Decomposition in @choleskyMatrices of 
* Covariance Matrix and u is a vector with the corresponding dimensions that has components samples from the standart normal distribution
* */
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

