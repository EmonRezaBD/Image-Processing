
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

void writeVecInCSV(vector<int> & dataVector) {
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

void writeCSV(string filename, cv::Mat m)
{
    ofstream myfile;
    myfile.open(filename.c_str());
    myfile << cv::format(m, cv::Formatter::FMT_CSV) << std::endl;
    myfile.close();
}

void ImageReadOpenCv(string filename)
{
    cv::Mat img = cv::imread(filename);
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    //Canny filter
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150, 3); //3 is aparture
    //writeCSV("CannyEdgeC++.csv",edges);

}

int main()
{
    std::cout << "Hello World!\n";


    //std::string readLocation = "../img/14264320.BMP";
    //std::string writeLocation = "../img/14264320Canny.BMP";
    double lowerThreshold = 0.03; //it can be changed
    double higherThreshold = 0.1;

    string filename = "../img/14264320.BMP";
    string writeFile = "../img/";
    //ImageReadOpenCv(filename);
    cv::Mat img = cannyEdgeDetection(filename, writeFile, 50, 150);
    writeCSV("RawCannyC++.csv", img);
    
    return 0;

}

