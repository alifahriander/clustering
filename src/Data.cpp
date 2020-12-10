#include <fstream>
#include <cmath>
#include "Data.h"
#include "assert.h"
#include "Trainer.h"

Data::Data(Observation inputObservation, double R_z, unsigned random_seed){
    generator_data.seed(random_seed);

    // Get numbers of clusters and samples 
    x_true = inputObservation.x;
    y_observed = inputObservation.y;
    assignments = inputObservation.assignments;

    int inSuccess = VectorToCSV(x_true, "/home/ander/Documents/git/clustering/x_true.csv", 4);
    inSuccess = VectorToCSV(y_observed, "/home/ander/Documents/git/clustering/y_observed.csv", 4);
    inSuccess = VectorToCSV(assignments, "/home/ander/Documents/git/clustering/assignments.csv", 4);

    numberClusters = x_true.rows();
    numberSamples = y_observed.rows();

    forwardMessageW = VectorXd(numberClusters);
    forwardMessageEta =VectorXd(numberClusters);

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
    r_z = R_z;

    s_z = VectorXd(numberClusters*numberSamples);
    for(unsigned int i=0; i<numberClusters*numberSamples; ++i){
        s_z(i) = Data::uniformDistribution(0, 1);
    }
    s_z = s_z.array().square();
    
    // Form W_x and W_z matrices 
    s_x = VectorXd(numberClusters);
    W_x = MatrixXd(numberClusters,numberClusters);


    W_z = MatrixXd(numberClusters*numberSamples, numberClusters*numberSamples);
    VectorXd V_z = s_z.array()+ r_z*r_z;
    VectorXd vectorW_z = V_z.array().inverse();
    W_z = vectorW_z.asDiagonal();


    //KMeans++ initialization
    x_estimate = initXEstimate();
    // Prepare z
    z = A*x_estimate - y;

    // Data::printData(); 
    Data::saveData(true);
   

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

void Data::updateCost(VectorXd v, double r, double& cost){
    double accumulatedCost = 0.0;
    for(unsigned int i=0; i<v.size(); i++){
        if(v(i)*v(i) < r*r) accumulatedCost += (v(i)*v(i) /(2*r*r)) + log(r);
        else accumulatedCost += log(abs(v(i))) + 0.5;
    }
    
    cost = accumulatedCost;
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

VectorXd Data::initXEstimate(){
    VectorXd centers(numberClusters);
    // Step 1 : Select one point from y as cluster center
    centers(0) = y_observed((unsigned int)uniformDistribution(0,numberSamples));
    cout << "Center 0:" << centers(0) << endl;
    double prevCenter = centers(0);

    for(unsigned int i = 1; i < numberClusters; i++){
        //Step 2 : Compute distances between 
        VectorXd distances = computeDistances(prevCenter);
        prevCenter = y(selectFromDistribution(distances));
        centers(i) = prevCenter;
        cout << "Center "<< i << ": " << centers(i) << endl;

    }
    return centers;

}

VectorXd Data::computeDistances(double center){
    VectorXd dists(numberSamples);
    for(unsigned int j=0; j<numberSamples; j++){
        dists(j) = (double) pow((y(j) - center),2);
    }
    return dists;

}

unsigned int Data::selectFromDistribution(VectorXd distances){
    distances = distances.normalized();
    // cout << distances << endl;
    double randomValue = uniformDistribution(0,1);
    double runningSum = 0.0;
    for(unsigned int i = 0; i<numberSamples; i++){
        runningSum += distances(i);
        if(runningSum > randomValue){
            return i;
        }
    }
    return 0;
}


void Data::printData(){
    cout << "========================================" << endl;
    cout << "S_x:" << endl << s_x << endl;
    cout << "S_z:" << endl << s_z << endl;
    cout << "x_estimate" << endl << x_estimate << endl;
    cout << "z=(Ax-y):" << endl << z << endl;


}

void Data::updateData(){
    Data::updateW(W_z, s_z, r_z);
    //TODO: Add updateCost
    // Data::printData();
    Data::saveData(false);
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
int Data::VectorToCSV(const double Scalar, const string& fileName, const streamsize dPrec){

	ofstream outputData;
    // Always appends
	outputData.open(fileName, ios::app);
	if (!outputData)
		return -1;
	outputData.precision(dPrec);
    outputData << Scalar;
    outputData << endl;
	outputData.close();

	if (!outputData)
		return -1;
	return 0;
}


void Data::saveData(bool init){
    int success = Data::VectorToCSV(x_estimate,"/home/ander/Documents/git/clustering/x_estimate.csv", 4);
    if(success != 0) cout<<"Vector couldn't be saved!"<<endl;
    success = Data::VectorToCSV(s_x,"/home/ander/Documents/git/clustering/s_x.csv", 4);
    if(success != 0) cout<<"Vector couldn't be saved!"<<endl;
    success = Data::VectorToCSV(s_z,"/home/ander/Documents/git/clustering/s_z.csv", 4);
    if(success != 0) cout<<"Vector couldn't be saved!"<<endl;
    success = Data::VectorToCSV(z,"/home/ander/Documents/git/clustering/z.csv", 4);
    if(success != 0) cout<<"Vector couldn't be saved!"<<endl;
    if(!init){
    success = Data::VectorToCSV(costZ,"/home/ander/Documents/git/clustering/costZ.csv", 10);
    if(success != 0) cout<<"Vector couldn't be saved!"<<endl;
    }
}

