#include "util.h"
using namespace std;
using namespace Eigen;

/**
 * 
 * Write @inputMatrix into a csv file with @fileName with the precision @dPrec
 * 
 * **/
int writeMatrix(const MatrixXd& inputMatrix, const string& fileName, const streamsize dPrec){
    int i, j;
    ofstream outputData;
    outputData.open(fileName, ios::app);
    if(!outputData) return -1;
    outputData.precision(dPrec);
    for(i=0; i < inputMatrix.rows(); i++){
        for(j = 0; j < inputMatrix.cols(); j++){
            outputData << inputMatrix(i,j);
            if(j < inputMatrix.cols()-1)
                outputData << ",";
        }
        if(i < inputMatrix.rows())
            outputData << endl;
    }
    outputData.close();
    if(!outputData) return -1;
    return 0;
}

/**
 * 
 * Read @inputMatrix from a csv file with @fileName with the precision @dPrec
 * 
 * **/
int readMatrix(MatrixXd& outputMatrix, const string& fileName, const streamsize dPrec){
    ifstream inputData;
    inputData.open(fileName);
    inputData.precision(dPrec);
    if(!inputData) return -1;
    cout << "File opened" << endl;
    string fileLine, fileCell;
    unsigned int prevNCols =0, nRows=0, nCols = 0;
    while(getline(inputData, fileLine)){
        nCols = 0;
        stringstream  linestream(fileLine);
        while(getline(linestream, fileCell, ',')){
            try{
                stod(fileCell);
            }catch(...){
                return -1;
            }
            nCols++;
        }
        
        if(nRows++ == 0) prevNCols = nCols;
        // For rows with different number of columns
        if(prevNCols != nCols) return -1;
    } 
    cout << "Dimensions are fetched" << endl;
    inputData.close();
    outputMatrix.resize(nRows, nCols);
    cout << "output matrix is resized" << endl;
    inputData.open(fileName);
    nRows = 0;
    cout << "reading matrix" << endl;
    while(getline(inputData, fileLine)){
        nCols = 0;
        stringstream linestream(fileLine);
        while(getline(linestream, fileCell, ',')){
            outputMatrix(nRows, nCols++) = stod(fileCell);
        }
        nRows++;
    }
    cout << "Matrix read" << endl;
    return 0;

}