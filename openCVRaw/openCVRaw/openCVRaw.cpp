
#include <iostream>
#include <fstream>

#include "canny.h"

//For testing
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;

void writeVecInCSV(vector<int> &dataVector) {
    // Open the CSV file for writing
    std::ofstream outputFile("output.csv");

    // Check if the file is opened successfully
    if (!outputFile.is_open()) {
        std::cerr << "Error: Could not open the file." << std::endl;
        return;
    }

    // Write vector elements to the CSV file
    for (size_t i = 0; i < dataVector.size(); ++i) {
        outputFile << dataVector[i];

        // Add a comma after each element except the last one
        if (i < dataVector.size() - 1) {
            outputFile << ",";
        }
    }

    // Close the file
    outputFile.close();

    std::cout << "CSV file written successfully." << std::endl;
}

int main()
{
    std::cout << "Hello World!\n";


    std::string readLocation = "../img/14264320.BMP";
    std::string writeLocation = "../img/14264320Canny.BMP";
    double lowerThreshold = 0.03; //it can be changed
    double higherThreshold = 0.1;

    //std::vector<int> vec = cannyEdgeDetection(readLocation, writeLocation, lowerThreshold, higherThreshold);
    //writeVecInCSV(vec);

    cv::Canny(gray, edges, 50, 150, 3);
    //Tmorrow--1. try cv::canny then try the raw c++
    
    
    return 0;

}

