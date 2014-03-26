//============================================================================
// Name        : CSSMatting.cpp
// Author      : Henri Rebecq
// Description : Implementation of "Improving image matting with comprehensive sample sets"
//============================================================================

#include "CSSMatting.h"

using namespace cv;
using namespace std;

/** Global variables */
Mat input, trimap;
int w,h;
vector<Region> regionsFG, regionsBG; // FG and BG are divided into overlapping regions
Region fg, bg, uk; // Each zone (FG,BG,Unknown) is stored as an instance of class Region
vector<vector<CandidateSample> > list_cdd_fg, list_cdd_bg; // Each overlapping region has its own sample set
Mat distanceMapUK, binaryMapUK;
float ukMaxDist = 0.f;

/** Parameters of the algorithm to tweak */
int TYPE_OPTIM = 1; // which objective function is maximized when selecting best (F,B) pair?
int nRegions = 4; // number of subregions
int nColorClusters = 4; // number of clusters for the first subregion (grows with the size of the region)
int nSpatialClusters = 4;
int covMatType = EM::COV_MAT_SPHERICAL; // or EM::COV_MAT_DIAGONAL
double resizeFactor = 1.0; // zoom display if input image is too small

int main(int argc, char* argv[]) {

	srand(time(NULL));

	string imgstr = "koala_bear.png"; // default image to be loaded
	if(argc == 2)
		imgstr = argv[1];
	input = imread("input/" + imgstr,1);
	trimap = imread("trimap/" + imgstr,0); // uchar

	cout << "Starting matting with image " << imgstr << endl;

	input.convertTo(input,CV_64FC3,1.0/255.0);
	w = input.cols;
	h = input.rows;

	buildTriRegions(trimap); // instantiate regions for FG and BG
	prepareAlphaMatte(); // most work is done here: regions are clustered and sampling sets are computed
	Mat alphaMatte = updateAlphaMatte(); // for each unknown point, select the best (F,B) pair and compute alpha

	initInteractiveWindows();
	display("Input + (F,B)",input,0);
	display("Alpha Matte",alphaMatte,0);
	display("Sample set",trimap,0);

	alphaMatte.convertTo(alphaMatte, CV_8UC3,255);
	imwrite("matte_"+imgstr, alphaMatte);

	waitKey(0);

	return 0;
}

/** Set up the graphical environment for interactive display of matte + sample sets */
void initInteractiveWindows() {
	namedWindow("Input + (F,B)", CV_WINDOW_AUTOSIZE );
	createTrackbar("Energy type", "Input + (F,B)", &TYPE_OPTIM, 4, updateObjectiveType);
	namedWindow("Alpha Matte", CV_WINDOW_AUTOSIZE );
	namedWindow("Sample set", CV_WINDOW_AUTOSIZE );
}

/** Builds an instance of region for FG, BG, and Unknown region */
void buildTriRegions(const Mat & trimap) {
	vector<Point> fgPoints, bgPoints, ukPoints;
	for(int x=0;x<w;x++) {
		for(int y=0;y<h;y++) {
			if(trimap.at<uchar>(y,x) == 0) bgPoints.push_back(Point(x,y));
			else if(trimap.at<uchar>(y,x) == 255) fgPoints.push_back(Point(x,y));
			else ukPoints.push_back(Point(x,y));
		}
	}
	fg = Region(fgPoints), bg = Region(bgPoints), uk = Region(ukPoints);
}

/** Splits a given region into small overlapping regions whose width grow quadratically until region is filled */
vector<Region> split(Region & region) {
	vector<Region> subRegions;
	Mat originalMap = region.getBinaryMap();
	Mat distanceMap;
	// Compute region's max width (max of the distance transform)
	float maxDist = 0.f;
	distanceTransform(originalMap,distanceMap,CV_DIST_L2,3);
	for(int x=0;x<w;x++) {
		for(int y=0;y<h;y++) {
			if(distanceMap.at<float>(y,x) > maxDist)
				maxDist = distanceMap.at<float>(y,x);
		}
	}
	// Each region should contain all points whose distance from the border is smaller than its width
	for(int k=1;k<=nRegions;k++) {
		vector<Point> pointList;
		float currentWidth = pow((float)k/(float)nRegions,2) * maxDist;
		for(int x=0;x<w;x++) {
			for(int y=0;y<h;y++) {
				if((originalMap.at<uchar>(y,x) > 0) && (distanceMap.at<float>(y,x) <= currentWidth))
					pointList.push_back(Point(x,y));
			}
		}
		subRegions.push_back(pointList);
	}
	return subRegions;
}

/** Computes the clusters and candidate sample points for each cluster
 * Needs to be called before any call to updateAlphaMatte() */
void prepareAlphaMatte() {
	// computes the max distance of two points in the unknown zone. is used later in function findRegionIndex(z)
	binaryMapUK = uk.getBinaryMap();
	distanceTransform(binaryMapUK,distanceMapUK,CV_DIST_L2,3);
	for(int x=0;x<w;x++) {
		for(int y=0;y<h;y++) {
			if(distanceMapUK.at<float>(y,x) > ukMaxDist)
				ukMaxDist = distanceMapUK.at<float>(y,x);
		}
	}
	// split FG and BG into small overlapping regions whose width grows quadratically until region is filled
	regionsFG = split(fg);
	regionsBG = split(bg);
	// for each splitted region generate its own candidate samples
	for(int i=0;i<nRegions;i++) {
		int nC = ceil(nColorClusters*pow(i+1,0.33));
		int nS = ceil(nSpatialClusters*pow(i+1,0.33));
		cout << "Clustering subregion " << (i+1) << " of zone FG ";
		cout << "(" << nC << " color clusters, ";
		cout << nS << " spatial clusters)" << endl;
		list_cdd_fg.push_back(generateSamplingSet(regionsFG[i],nC,nS));
	}
	for(int i=0;i<nRegions;i++) {
		int nC = ceil(nColorClusters*pow(i+1,0.33));
		int nS = ceil(nSpatialClusters*pow(i+1,0.33));
		cout << "Clustering subregion " << (i+1) << " of zone BG ";
		cout << "(" << nC << " color clusters, ";
		cout << nS << " spatial clusters)" << endl;
		list_cdd_bg.push_back(generateSamplingSet(regionsBG[i],nC,nS));
	}
}

/** Computes the alpha matte */
Mat updateAlphaMatte() {
	cout << "Updating alpha matte..." << endl;
	Mat alphaMatte = Mat::zeros(Size(w,h),CV_64F);
	for(int p=0;p<uk.getN();p++) {
		Point z = uk.getPoint(p);
		int indexRegion = findRegionIndex(z);
		vector<CandidateSample> cdd_fg = list_cdd_fg[indexRegion];
		vector<CandidateSample> cdd_bg = list_cdd_bg[indexRegion];
		Point optimalIndices = maximizeEnergyOverPairs(z);

		Vec3d Iz = input.at<Vec3d>(z);
		Vec3d Fi = cdd_fg[optimalIndices.x].getColor();
		Vec3d Bj = cdd_bg[optimalIndices.y].getColor();
		double alpha = (Iz-Bj).dot(Fi-Bj)/pow(norm(Fi-Bj),2);
		alphaMatte.at<double>(z) = max(min(alpha,1.0),0.0);
	}
	for(int p=0;p<fg.getN();p++) alphaMatte.at<double>(fg.getPoint(p)) = 1.0f;
	for(int p=0;p<bg.getN();p++) alphaMatte.at<double>(bg.getPoint(p)) = 0.0f;

	cout << "Done" << endl;

	return alphaMatte;
}

/** Given a region, generate all candidate samples using two-level hierarchical clustering */
vector<CandidateSample> generateSamplingSet(const Region & region, int nColorClusters, int nSpatialClusters) {
	vector<CandidateSample> cdd_samples;

	// cluster region using color
	vector<Region> colorClusters = cluster(nColorClusters, region, 0);
	// for each color cluster
	for(int k=0;k<nColorClusters;k++) {
		// cluster it into spatial sub-clusters
		vector<Region> spatialClusters = cluster(nSpatialClusters, colorClusters[k], 1);
		// update list of candidate points (centroids of the subclusters)
		for(int l=0;l<nSpatialClusters;l++) {
			Region * temp = new Region(spatialClusters[l]);
			cdd_samples.push_back(CandidateSample(temp, temp->getBarycenter(), temp->getMeanColor()));
		}
	}
	return cdd_samples;
}

/** Takes as input a region to cluster, cluster it wrt color (resp. spatial coordinate)
 *  and return a list of clustered regions.
 */
vector<Region> cluster(const int nClusters, const Region & region, const short typeClustering) {
	int N = region.getN();
	vector<Region> clusters;
	Mat samples;

	// Pre-process data as OpenCV's EM algorithm requires a one-channel CV_64F matrix
	switch(typeClustering) {
	case 0: // COLOR
		samples = Mat(N,3,CV_64FC1);
		for(int i=0;i<N;i++) {
			for(int c=0;c<3;c++) samples.at<double>(i,c) = region.getColor(i)[c];
		}
		break;
	case 1: // SPATIAL
		samples = Mat(N,2,CV_64FC1);
		for(int i=0;i<N;i++) {
			samples.at<double>(i,0) = (double)region.getPoint(i).x;
			samples.at<double>(i,1) = (double)region.getPoint(i).y;
		}
		break;
	}

	// Perform clustering using EM
	EM model(nClusters,covMatType);
	Mat probs, log_likelihoods, labels;
	model.train(samples,log_likelihoods,labels,probs);
	for(int k=0;k<nClusters;k++) {
		vector<Point> tempList;
		for(int i=0;i<N;i++) {
			if(labels.at<int>(i,0) == k) tempList.push_back(region.getPoint(i));
		}
		Region reg(tempList);
		clusters.push_back(reg);
	}
	return clusters;
}

/** Distortion between estimated color and observed color */
double Kz(const Vec3d & Iz, const Vec3d & Fi, const Vec3d & Bi) {
	double alpha = (Iz-Bi).dot(Fi-Bi)/pow(norm(Fi-Bi),2);
	alpha = max(min(alpha,1.0),0.0);
	Vec3d Iz_est = alpha*Fi + (1-alpha)*Bi;
	return exp(-norm(Iz-Iz_est));
}

/** Normalization constant used in Cz */
double maxCohenDistance(const vector<CandidateSample> & cdd_fg, const vector<CandidateSample> & cdd_bg) {
	double max_Cz = -1.0, C = 0.0;
	for(unsigned int i=0;i<cdd_fg.size();i++) {
		for(unsigned int j=0;j<cdd_bg.size();j++) {
			C = Cz(cdd_fg[i],cdd_bg[j]);
			if(C > max_Cz)
				max_Cz = C;
		}
	}
	return max_Cz;
}

/** Normalization constant used in Sz. Separated for efficiency reasons */
double sumDist(const Point & z, const vector<CandidateSample> & cdd) {
	double sumDist = 0.0;
	for(unsigned int i=0;i<cdd.size();i++) {
		sumDist += norm(z-cdd[i].getPos());
	}
	return sumDist;
}

/** Distance between (F,B) pair */
double Sz(const Point & z, const Point & posFi, const Point & posBi, double sumDistFG, double sumDistBG) {
	if(sumDistBG == 0.0 || sumDistFG == 0.0)
		return 0.0;
	else
		return exp(-norm(z-posFi)/sumDistFG) * exp(-norm(z-posBi)/sumDistBG);
}

/** Inverse overlap between distributions that generated Fi and Bi */
double Cz(const CandidateSample & Fi, const CandidateSample & Bi) {
	Region *dFi = Fi.getSourceRegion(), *dBi = Bi.getSourceRegion();
	Vec3d meanFi = dFi->getMeanColor(), meanBi = dBi->getMeanColor();
	//	return norm(meanBi-meanFi);

	int NFi = dFi->getN(), NBi = dBi->getN();
	Vec3d varFi = dFi->getVar(), varBi = dBi->getVar();
	double csq = 1.0;
	for(int c=0;c<3;c++) {
		if(NFi+NBi-2 <= 0)
			return 0.0;
		csq += pow((meanFi[c]-meanBi[c]) / sqrt(((NFi-1)*varFi[c] + (NBi-1)*varBi[c])/(NFi+NBi-2)),2);
	}
	return sqrt(csq);
}

double max(double a, double b) {
	return (a > b) ? a : b;
}

/* Return (i,j), the indices of the best (F,B) pair in lists cdd_fg and cdd_bg */
Point maximizeEnergyOverPairs(Point & z) {
	int indexRegion = findRegionIndex(z);
	vector<CandidateSample> cdd_fg = list_cdd_fg[indexRegion];
	vector<CandidateSample> cdd_bg = list_cdd_bg[indexRegion];
	double maxObjective = -1.0, S = 0.0, K = 0.0, C = 0.0, objective = 0.0;
	double sumFGDist = sumDist(z, cdd_fg), sumBGDist = sumDist(z, cdd_bg);
	double max_Cz = maxCohenDistance(cdd_fg, cdd_bg);
	int best_i = 0, best_j = 0;
	for(unsigned int i=0;i<cdd_fg.size();i++) {
		for(unsigned int j=0;j<cdd_bg.size();j++) {
			S = Sz(z, cdd_fg[i].getPos(),cdd_bg[j].getPos(),sumFGDist,sumBGDist);
			K = Kz(input.at<Vec3d>(z),cdd_fg[i].getColor(),cdd_bg[j].getColor());
			C = Cz(cdd_fg[i],cdd_bg[j]) / max_Cz;
			switch(TYPE_OPTIM) {
			case 0:
				objective = K;
				break;
			case 1:
				objective = K*S;
				break;
			case 2:
				objective = K*S*C;
				break;
			case 3:
				objective = S;
				break;
			case 4:
				objective = C;
				break;
			}
			if(objective > maxObjective) {
				maxObjective = objective;
				best_i = i;
				best_j = j;
			}
		}
	}
	return Point(best_i, best_j);
}

/** Given a point z belonging to the unknown region, in which subregion is it ? */
int findRegionIndex(Point & z) {
	for(int k=1;k<=nRegions;k++) {
		vector<Point> pointList;
		float borderDist = pow((float)k/(float)nRegions,2) * ukMaxDist;
		if(distanceMapUK.at<float>(z) <= borderDist)
			return (k-1);
	}
	return 0;
}

/** Displays the given image in a window */
void display(const string & title, Mat & img, const int stop) {
	Mat imgResized;
	resize(img,imgResized,Size(),resizeFactor,resizeFactor,INTER_LINEAR);
	setMouseCallback(title, updateDisplay, NULL);
	imshow( title, imgResized);
	if(stop)
		waitKey(0);
}

/** Update the alpha matte when the type of energy is changed */
void updateObjectiveType(int, void*) {
	cout << "Updated objective function to : O = " + typeOptimStr[TYPE_OPTIM] << endl;
	Mat alphaMatte = updateAlphaMatte();
	display("Alpha Matte",alphaMatte,0);
}

/** Handles interactive display of best (F,B) pair for any point */
void updateDisplay(int event, int x, int y, int flags, void* userData) {
	int radius = 3;
	if  (event == EVENT_LBUTTONDOWN) {
		Point z(x,y);
		z.x /= resizeFactor;
		z.y /= resizeFactor;
		Mat csamples, dispRegions = Mat::zeros(Size(w,h),CV_64FC3);
		input.copyTo(csamples);

		// if clicked point is not in the unknown zone, do nothing
		if(binaryMapUK.at<uchar>(z) == 0)
			return;

		int indexRegion = findRegionIndex(z);
		vector<CandidateSample> cdd_fg = list_cdd_fg[indexRegion];
		vector<CandidateSample> cdd_bg = list_cdd_bg[indexRegion];

		// find best (F,B) pair to explain point z
		Point optimalIndices = maximizeEnergyOverPairs(z);
		Vec3d Iz = input.at<Vec3d>(z);
		Vec3d Fi = cdd_fg[optimalIndices.x].getColor();
		Vec3d Bj = cdd_bg[optimalIndices.y].getColor();
		double alpha = (Iz-Bj).dot(Fi-Bj)/pow(norm(Fi-Bj),2);
		alpha = max(min(alpha,1.0),0.0);
		cout << "alpha = " << alpha << endl;
		Scalar colorLine = blue + alpha*Scalar(-1.0,-0.5,1.0);

		// update display of current candidate regions and points
		for(unsigned int p=0;p<cdd_fg.size();p++) {
			(cdd_fg[p].getSourceRegion())->drawRegion(dispRegions, red, true);
			(cdd_bg[p].getSourceRegion())->drawRegion(dispRegions, blue, true);
		}
		circle(dispRegions,z,radius,yellow,-1);
		line(dispRegions, z, cdd_fg[optimalIndices.x].getPos(), colorLine);
		line(dispRegions, z, cdd_bg[optimalIndices.y].getPos(), colorLine);

		// update display of candidate points along with input image
		circle(csamples,z,radius,yellow,-1);
		circle(csamples,cdd_fg[optimalIndices.x].getPos(),radius,red,-1);
		circle(csamples,cdd_bg[optimalIndices.y].getPos(),radius,blue,-1);
		line(csamples, z, cdd_fg[optimalIndices.x].getPos(), colorLine);
		line(csamples, z, cdd_bg[optimalIndices.y].getPos(), colorLine);

		display("Input + (F,B)",csamples,0);
		display("Sample set",dispRegions,0);
	}
}
