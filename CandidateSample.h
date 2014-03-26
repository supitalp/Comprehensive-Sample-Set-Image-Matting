/*
 * CandidateSample.h
 *
 *  Created on: 12 mars 2014
 *      Author: henri
 */

#ifndef CANDIDATESAMPLE_H_
#define CANDIDATESAMPLE_H_

#include "Region.h"

class CandidateSample {
public:
	CandidateSample(Region * _sourceRegion, cv::Point _pos, cv::Vec3d _color);
	virtual ~CandidateSample();

	const cv::Point& getPos() const {
		return pos;
	}

//	const cv::Vec3d& getColor() const {
//		return input.at<cv::Vec3d>(pos);
//	}

	Region* getSourceRegion() const {
		return sourceRegion;
	}

	const cv::Vec3d& getColor() const {
		return color;
	}

private:
	Region * sourceRegion;
	cv::Point pos;
	cv::Vec3d color;
};

#endif /* CANDIDATESAMPLE_H_ */
