/*
 * Region.cpp
 *
 *  Created on: 12 mars 2014
 *      Author: Henri Rebecq
 */

#include "Region.h"

using namespace cv;
using namespace std;

Region::Region() {

}

Region::~Region() {
}

/** Build a region containing all non-zero pixels */
Region::Region(Mat & binaryImage) : w(input.cols), h(input.rows) {
	assert(binaryImage.cols == input.cols);
	assert(binaryImage.rows == input.rows);
	assert(binaryImage.type() == CV_8U);
	vector<Point> pointList;
	for(int x=0;x<w;x++) {
		for(int y=0;y<h;y++) {
			if(binaryImage.at<uchar>(y,x) > 0) pointList.push_back(Point(x,y));
		}
	}
	points = pointList;
	N = points.size();
	init();
}

Region::Region(vector<Point> _points) : points(_points), N(_points.size()), w(input.cols), h(input.rows) {
	init();
}

void Region::init() {
	// Compute barycenter
	barycenter = Point(0,0);
	var = 0.0;
	for(int i=0;i<N;i++) {
		barycenter += points[i];
		meanColor += input.at<Vec3d>(points[i]);
	}
	barycenter.x = floor((double)barycenter.x / N);
	barycenter.y = floor((double)barycenter.y / N);
	meanColor /= (double)N;

	// Compute variance
	for(int i=0;i<N;i++) {
		const Vec3d std = input.at<Vec3d>(points[i])-meanColor;
		var += std.mul(std);
	}
	var /= (double)(N-1);

	// Compute region's binary map
	binaryMap = Mat::zeros(Size(w,h),CV_8U);
	for(int i=0;i<N;i++) {
		binaryMap.at<uchar>(points[i]) = 255;
	}
}

void Region::drawRegion(cv::Mat & img, const Scalar & borderColor, bool drawCandidates) {
	assert(img.type() == CV_64FC3);
	assert(img.rows == h && img.cols == w);
	Vec3d color = meanColor;
	//	Vec3d color = Vec3d((double)rand() / RAND_MAX, (double)rand() / RAND_MAX, (double)rand() / RAND_MAX);
	//	Vec3d color = 50.0*Vec3d(norm(var),norm(var),norm(var));
	//	Vec3d color = 10.0*Vec3d(sqrt(var[0]),sqrt(var[1]),sqrt(var[2]));
	for(int i=0;i<N;i++) {
		img.at<Vec3d>(points[i]) = color;
		// img.at<Vec3d>(points[i]) = Vec3d(borderColor[0],borderColor[1],borderColor[2]);
	}
	if(drawCandidates) {
//		circle(img,barycenter,4,borderColor,-1);
		circle(img,barycenter,3,(Scalar)meanColor,-1);
		circle(img,barycenter,4,borderColor,1,CV_AA);
	}
}
