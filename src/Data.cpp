#include <fstream>
#include "Data.h"
#include "assert.h"

Data::Data(Observation inputObservation, double R_x, double R_z, unsigned random_seed){
    generator_data.seed(random_seed);
    // Get numbers of clusters and samples 
    x_true = inputObservation.x;
    y_observed = inputObservation.y;

    int inSuccess = VectorToCSV(x_true, "/home/ander/Documents/git/clustering/x_true.csv", 4);
    inSuccess = VectorToCSV(y_observed, "/home/ander/Documents/git/clustering/y_observed.csv", 4);
    inSuccess = VectorToCSV(inputObservation.assignments, "/home/ander/Documents/git/clustering/assignments.csv", 4);

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
    r_x = R_x;
    r_z = R_z;

    s_x = VectorXd(numberClusters);
    for(unsigned int i=0; i<numberClusters; ++i){
        s_x(i) = Data::uniformDistribution(0, 1);
    }
    s_x = s_x.array().square();

    s_z = VectorXd(numberClusters*numberSamples);
    for(unsigned int i=0; i<numberClusters*numberSamples; ++i){
        s_z(i) = Data::uniformDistribution(0, 1);
    }
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
    // x_estimate = VectorXd::Constant(numberClusters,0.0);
    x_estimate = VectorXd(numberClusters);
    // x_estimate << -1.0, 1.0;
    for(unsigned int i=0; i<numberClusters; ++i){
        x_estimate(i) = Data::normalDistribution(0, V_x(i));
    }
    
    // Prepare z
    z = A*x_estimate - y;

    // Data::printData(); 
    Data::saveData();
   

}

/*
* Update Rule for W by updating s_x(i)^2 according to (14.25) in lecture notes 
* @param W
* @param s
* @param r: fix initial variance 
*/
void Data::updateW(MatrixXd& W,VectorXd s, double r){
    VectorXd sumVariance = s.array() + r*r;
    sumVariance = sumVariance.array().inverse();
    W = sumVariance.asDiagonal();
}


/*
* Draw sample from normal distribution
* @param mean
* @param variance
* @return sample 
*/
double Data::normalDistribution(double mean, double variance){
    normal_distribution<double> distribution(mean, variance);
    return distribution(generator_data);
}
double Data::uniformDistribution(double min, double_t max){
    uniform_real_distribution<double> distrib(min, max); 
    return distrib(generator_data);

}

void Data::printData(){
    cout << "========================================" << endl;
    cout << "S_x:" << endl << s_x << endl;
    cout << "S_z:" << endl << s_z << endl;
    cout << "x_estimate" << endl << x_estimate << endl;
    cout << "z=(Ax-y):" << endl << z << endl;


}

void Data::updateData(){
    Data::updateW(W_x, s_x, r_x);
    Data::updateW(W_z, s_z, r_z);
    // Data::printData();
    Data::saveData();
}


int Data::VectorToCSV(const MatrixXd& inputMatrix, const string& fileName, const streamsize dPrec) {
	int i;
    assert( inputMatrix.cols() == 1 );

	ofstream outputData;
    // Always appends
	outputData.open(fileName, ios::app);
	if (!outputData)
		return -1;
	outputData.precision(dPrec);
	for (i = 0; i < inputMatrix.rows(); i++) {
		outputData << inputMatrix(i);
        if(i<inputMatrix.rows()-1) outputData << ",";
        else outputData<<endl;
	}
	outputData.close();

	if (!outputData)
		return -1;
	return 0;
}


void Data::saveData(){
    int success = Data::VectorToCSV(x_estimate,"/home/ander/Documents/git/clustering/x_estimate.csv", 4);
    if(success != 0) cout<<"Vector couldn't be saved!"<<endl;
    success = Data::VectorToCSV(s_x,"/home/ander/Documents/git/clustering/s_x.csv", 4);
    if(success != 0) cout<<"Vector couldn't be saved!"<<endl;
    success = Data::VectorToCSV(s_z,"/home/ander/Documents/git/clustering/s_z.csv", 4);
    if(success != 0) cout<<"Vector couldn't be saved!"<<endl;
    success = Data::VectorToCSV(z,"/home/ander/Documents/git/clustering/z.csv", 4);
    if(success != 0) cout<<"Vector couldn't be saved!"<<endl;

}

