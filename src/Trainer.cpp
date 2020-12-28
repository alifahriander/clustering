#include <fstream>
#include "Trainer.h"
#include "Data.h"

using namespace std;
using namespace Eigen;

Trainer::Trainer(unsigned int inputNumberInterations, double inputTol){
    Trainer::numberIterations = inputNumberInterations;
    Trainer::tol = inputTol;
    
}


/**
 * Update x_estimate
 * */
void Trainer::updateX(Data& data){ 
    //Update x_estimate with forward and backward Gaussian message
    for(int k = 0; k<data.numberClusters*data.dimension; k++){
        double w = 0.0;
        double eta = 0.0;
        double tmp = 0.0;
        
        // Backward Message
        for(int i=0;i<data.numberSamples;i++){
            // Consider s_z if it is not equal to zero
            tmp = data.r_z * data.r_z + data.s_z(i*data.numberClusters*data.dimension + k);
            w += 1 / tmp ;
            eta+= (1/tmp) * data.y(k+i*data.numberClusters*data.dimension);
        }

        // Forward Message
        w += data.forwardMessageW(k);
        eta += data.forwardMessageEta(k);
        data.forwardMessageW(k) = w;
        data.forwardMessageEta(k) = eta;
        
        data.x_estimate(k) =  (1 - data.alpha) * data.x_estimate(k) + data.alpha * eta / (w);
        data.s_x(k) = 1/w;

    }
    data.z = data.A * data.x_estimate - data.y;
    
}


void Trainer::updateSz(Data& data){
    
    //diffZ = Vector(z_i ^2 - r_z^2)
    VectorXd diffZ = data.z.array().pow(2);
    diffZ -= data.r_z * data.r_z * VectorXd::Constant(diffZ.rows(),1.0);
    
    for(unsigned int i=0; i<data.numberClusters*data.numberSamples; ++i){

        if(diffZ(i) < 0.0) data.s_z(i) = 0;
        else data.s_z(i) = diffZ(i);
    }
    if(multiDimension){
        for(unsigned int i=0; i<data.numberSamples; i++){
            //Find max 
            for(unsigned int k=0; k<data.numberClusters; k++){
                double maxVariance = 0.0;

                for(unsigned int d=0; d<data.dimension; d++){
                    unsigned int index = i*data.numberClusters*data.dimension + k + d*data.numberClusters;
                    if(data.s_z(index)>maxVariance) maxVariance = data.s_z(index);
                }
                for(unsigned int d=0; d<data.dimension; d++){
                    unsigned int index = i*data.numberClusters*data.dimension + k + d*data.numberClusters;
                    data.s_z(index) = maxVariance;
                }
                // cout << maxVariance << endl;
            }
        }
    }    
}

/**
 * Training Loop
 * */
void Trainer::train(Data& data){
    if(data.dimension != 1) multiDimension = true;
    else multiDimension = false;
    
    double diff = 0.0;
    unsigned int noChangeCounter = 0;

    for(unsigned int counter=0; counter<Trainer::numberIterations; counter++){
        //Save last estimate for convergence condition
        Trainer::stateX = data.x_estimate;    
        Trainer::updateSz(data);
        Trainer::updateX(data);
        

        //If at least one of s_xi is 0, stop the training
        bool stopTraining = false;
        for(unsigned int i =0; i<data.numberClusters;i++){
            if(isinf(data.s_x(i))){
                stopTraining = true;
            }
        }

        diff = (stateX-data.x_estimate).norm(); 
        if(counter%50 == 0) cout << " Iteration: "<< counter << "\tDifference: "<< diff << endl;
        
        if(stopTraining) break;

        data.updateCost(data.z,data.r_z, data.costZ);
        data.saveData();



        if(diff< Trainer::tol){
            noChangeCounter++;
            if(noChangeCounter == 10){
                cout << "CONVERGENCE!" << endl;
                break;
            }
            
        }

    }
    
}
