#include "Data.h"


Data::Data(VectorXd x_true, VectorXd y_observed, double r, double s, double initX, unsigned random_seed){
    generator_data.seed(random_seed);
    // Get numbers of clusters and samples 
    numberClusters = x_true.rows();
    numberSamples = y_observed.rows();

    // Prepare y 
    y = VectorXd(numberSamples*numberClusters);
    for(unsigned int i=0; i<numberSamples; ++i){
        for(unsigned int j=0; j<numberClusters; ++j){
            y(i*numberClusters + j) = y_observed(i); 
        }
    }    
    
    // Prepare A
    A = MatrixXd(numberSamples*numberClusters, numberClusters);
    MatrixXd identity_matrix = MatrixXd::Identity(numberClusters, numberClusters);
    for(unsigned int i = 0; i<numberSamples; ++i){
        A.block(numberClusters*i,0,numberClusters,numberClusters) = identity_matrix;
    }

    // Prepare variances
    r_x = r;
    r_z = r;

    s_x = VectorXd::Constant(numberClusters,s);
    s_x = s_x.array().square();
    s_z = VectorXd::Constant(numberClusters*numberSamples,s);
    s_z = s_z.array().square();

    W_x = MatrixXd(numberClusters,numberClusters);
    VectorXd V_x = s_x.array() + r_x*r_x;
    VectorXd vectorW_x = V_x.array().inverse();
    W_x = vectorW_x.asDiagonal();

    W_z = MatrixXd(numberClusters*numberSamples, numberClusters*numberSamples);
    VectorXd V_z = s_z.array()+ r_z*r_z;
    VectorXd vectorW_z = V_z.array().inverse();
    W_z = vectorW_z.asDiagonal();

    // Draw x_estimate from normal distribution with mean zero and 
    // variance r_x^2 + s_x__i^2 
    x_estimate = VectorXd(numberClusters);
    for(unsigned int i=0; i<numberClusters; ++i){
        normal_distribution<double> norm_distribution(0.0, V_x(i));
        x_estimate(i) = norm_distribution(generator_data);
    }
    
    // Prepare z
    z = A*x_estimate - y;

    // Create a test 
    cout << "S_x:" << endl << s_x << endl;
    cout << "S_z:" << endl << s_z << endl;
    cout << "W_x:" << endl << W_x << endl;
    cout << "W_z:" << endl << W_z << endl;
    cout << "x_estimate" << endl << x_estimate << endl;
    cout << "Z:" << endl << z << endl;

}

void Data::updateW(MatrixXd& W,VectorXd s, double r){
    VectorXd sumVariance = s.array() + r*r;
    sumVariance = sumVariance.array().inverse();
    W = sumVariance.asDiagonal();
}

void Data::updateData(){
    Data::updateW(Data::W_x, Data::s_x, Data::r_x);
    Data::updateW(W_z, s_z, r_z);
    
    cout << "X ESTIMATE" << endl;
    cout << x_estimate << endl;

}

double Data::normalDistribution(double mean, double variance){
    normal_distribution<double> distribution(mean, variance);
    return distribution(generator_data);
}


