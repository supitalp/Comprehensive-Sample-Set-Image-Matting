/*
 * Region.h
 *
 *  Created on: 12 mars 2014
 *      Author: Henri Rebecq
 *
 */

#ifndef REGION_H_
#define REGION_H_

#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <opencv2/ml/ml.hpp>
#include <vector>
#include <time.h>

extern cv::Mat input;

/** This class represents a region (subset) of a given image.
 * It embeds a list of pixel positions (indexed over image 'input').
 */
class Region {
public:
	Region();
	Region(cv::Mat & binaryImage);
	Region(std::vector<cv::Point> _points);
	void init();
	virtual ~Region();

	void drawRegion(cv::Mat & img, const cv::Scalar & borderColor, bool drawCandidates);

	int getN() const {
		return N;
	}

	const cv::Vec3d getColor(int index) const {
		assert(index >= 0 && index < N);
		return input.at<cv::Vec3d>(points[index]);
	}

	const cv::Point getPoint(int index) const {
		assert(index >= 0 && index < N);
		return points[index];
	}

	const cv::Point& getBarycenter() const {
		return barycenter;
	}

	const cv::Vec3d& getMeanColor() const {
		return meanColor;
	}

	const cv::Vec3d getVar() const {
		return var;
	}

	const cv::Mat& getBinaryMap() const {
		return binaryMap;
	}

private:
	std::vector<cv::Point> points;
	int N; // number of points
	int w;
	int h;
	cv::Mat binaryMap;
	cv::Point barycenter;
	cv::Vec3d meanColor;
	cv::Vec3d var;
};

#endif /* REGION_H_ */
