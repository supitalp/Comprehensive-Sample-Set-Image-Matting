/*
 * CSSMatting.h
 *
 *  Created on: 19 mars 2014
 *      Author: henri
 */

#ifndef CSSMATTING_H_
#define CSSMATTING_H_

#include "Region.h"
#include "CandidateSample.h"

/* Prototypes of functions */
cv::Mat updateAlphaMatte();
void buildTriRegions(const cv::Mat & trimap);
std::vector<Region> split(Region & region);
void prepareAlphaMatte();
std::vector<CandidateSample> generateSamplingSet(const Region & region, int nColorClusters, int nSpatialClusters);
std::vector<Region> cluster(const int nClusters, const Region & region, const short typeClustering);
cv::Point maximizeEnergyOverPairs(cv::Point & z);
double Kz(const cv::Vec3d & Iz, const cv::Vec3d & Fi, const cv::Vec3d & Bi);
double Sz(const cv::Point & z, const cv::Point & posFi, const cv::Point & posBi, double sumDistFG, double sumDistBG);
double Cz(const CandidateSample & Fi, const CandidateSample & Bi);
double maxCohenDistance(const std::vector<CandidateSample> & cdd_fg, const std::vector<CandidateSample> & cdd_bg);
double sumDist(const cv::Point & z, const std::vector<CandidateSample> & cdd);
int findRegionIndex(cv::Point & z);
double max(double a, double b);

void initInteractiveWindows();
void display(const std::string & title, cv::Mat & img, const int stop);
void updateDisplay(int event, int x, int y, int flags, void* userData);
void updateObjectiveType(int, void*);

/* Constants */
const std::string typeOptimStr[5] = {"K","K.S","K.S.C","S","C"};
const cv::Scalar red(0.0,0.0,1.0);
const cv::Scalar blue(1.0,0.5,0.0);
const cv::Scalar yellow(0.0,1.0,1.0);


#endif /* CSSMATTING_H_ */
