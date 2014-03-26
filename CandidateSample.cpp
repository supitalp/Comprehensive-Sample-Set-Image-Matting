/*
 * CandidateSample.cpp
 *
 *  Created on: 12 mars 2014
 *      Author: henri
 */

#include "CandidateSample.h"

using namespace cv;
using namespace std;

CandidateSample::CandidateSample(Region * _sourceRegion, cv::Point _pos, cv::Vec3d _color) : sourceRegion(_sourceRegion), pos(_pos), color(_color) {
}

CandidateSample::~CandidateSample() {
}

