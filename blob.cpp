#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "dirent.h"
#include <iomanip>
#include <stdio.h>
#include <cmath>
#include <math.h>

using namespace cv;
using namespace std;


cv::SimpleBlobDetector detectorW;
cv::SimpleBlobDetector detectorB;
cv::SimpleBlobDetector::Params paramsW;
cv::SimpleBlobDetector::Params paramsB;
//void detectBlobN();
//void detectBlobP();
vector<vector<KeyPoint> > clusters;
void DBSCAN_keypoints(vector<KeyPoint> *keypoints, float eps, int minPts);//vector<vector<KeyPoint> >
vector<int> regionQuery(vector<KeyPoint> *keypoints, KeyPoint *keypoint, float eps);

int main(int argc, char* argv[])
{
	double t = (double)getTickCount();

    int i;
    int k;
    int done;
	double error = 0;
	double correct = 0;
    int falsePos = 0;
    int falseNeg = 0;

    // Change thresholds
	//paramsW.minRepeatability = 3;
    paramsW.minThreshold = 50;//50
    paramsW.maxThreshold = 220;//220
    paramsW.thresholdStep = 20;//20
	paramsW.minDistBetweenBlobs = 1;
    // Filter by Area.
    paramsW.filterByArea = true;
    paramsW.minArea = 256;//256
    paramsW.maxArea = 1028;//1028

    // Filter by Circularity
    //params.filterByCircularity = true;
    //params.minCircularity = 0.1;
    paramsW.filterByColor = true;
    paramsW.blobColor = 255;//255
    // Filter by Convexity
    paramsW.filterByConvexity = false;
    //paramsW.minConvexity = 0.9;

    // Filter by Inertia
    paramsW.filterByInertia = true;
    paramsW.minInertiaRatio = 0.1;//0.05
    paramsW.maxInertiaRatio = 0.5;//0.5


      // Set up detector with params
    cv::SimpleBlobDetector detectorW(paramsW);

    paramsB.minThreshold = 1;//150
    paramsB.maxThreshold = 210;//230
    paramsB.thresholdStep = 20;//20
    paramsB.minDistBetweenBlobs = 1;
    // Filter by Area.
    paramsB.filterByArea = true;
    paramsB.minArea = 128;//128
    paramsB.maxArea = 512;//512

    // Filter by Circularity
    //params.filterByCircularity = true;
    //params.minCircularity = 0.1;
    paramsB.filterByColor = true;
    paramsB.blobColor = 0;
    // Filter by Convexity
    paramsB.filterByConvexity = false;
    //paramsB.minConvexity = .01;

    // Filter by Inertia
    paramsB.filterByInertia = true;
    paramsB.minInertiaRatio = 0;//0.2
    paramsB.maxInertiaRatio = 1;//0.2

      // Set up detector with params
    cv::SimpleBlobDetector detectorB(paramsB);


    std::string inputDirectory = "/home/ryan/blobImage/images";

    DIR *directory = opendir (inputDirectory.c_str());
    struct dirent *_dirent = NULL;
    if(directory == NULL)
    {
        printf("Cannot open Input Folder\n");
        return 1;
    }
    while((_dirent = readdir(directory)) != NULL)
    {
        std::string fileName = inputDirectory + "/" +std::string(_dirent->d_name);
        char truthChar = fileName.at( fileName.length() - 5 );


        //std::cout << "The file name is: " << fileName << "\n";
		//std::cout << "truthChar is: " << truthChar << "\n";
        cv::Mat im = cv::imread(fileName.c_str(), IMREAD_GRAYSCALE);
        if(im.data == NULL)
        {
            printf("Cannot Open Image\n");
            continue;
        }
        // Add your any image filter here
        else{
            int whiteCount = 0;
            int blackCount = 0;
            //otsus method with adaptive threshold here
            //cv::adaptiveThreshold(im, OutputArray dst, double maxValue, int adaptiveMethod, int thresholdType, int blockSize, double C)¶
            //cv::Mat dst1;
            //cv::Mat dst2;
            //cv::adaptiveThreshold( im, dst1, 255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,13, 1 );
            //cv::threshold(im, dst2, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
            //cv::threshold(im, dst2, 0, 255, CV_THRESH_OTSU);
            std::vector<KeyPoint> keypointsW;
            detectorW.detect( im, keypointsW);
            std::vector<KeyPoint> keypointsB;
            detectorB.detect( im, keypointsB);
            std::vector<KeyPoint> keyPointsTotal;
            //std::cout << "size of keypointsW is: " << keypointsW.size() << "\n";
			//std::cout << "size of keypointsB is: " << keypointsB.size() << "\n";
            done = 0;
            if (keypointsW.size() > 0 && keypointsB.size() > 0){// && truthChar == 'P'){
                //std::vector<KeyPoint> keyPointsLabeled;
                //vector<vector<KeyPoint> > clusters;
                double avgX = 0;

                    for (i = 0; i < keypointsW.size(); i++){


                        keyPointsTotal.push_back(keypointsW.at(i));
                    }
                    for (k = 0; k < keypointsB.size(); k++){
                        keyPointsTotal.push_back(keypointsB.at(k));

                            //if ( hypot(keypointsW.at(i).pt.x - keypointsB.at(k).pt.x, keypointsW.at(i).pt.y - keypointsB.at(k).pt.y) < 50){
                                //std::cout << "keypointW = " << keypointsW.at(i).pt.x << "," <<keypointsW.at(i).pt.y << "\n";
                                //std::cout << "keypointB = " << keypointsB.at(k).pt.x << "," <<keypointsB.at(k).pt.y << "\n";

                    }
                    DBSCAN_keypoints(&keyPointsTotal, 40.0, 1);
                    //std::cout << "clusters count in main = " << clusters.size() << "\n";

                    if (clusters.size() > 1){
                        for ( int i = 1; i < 2; i ++){//the first cluster is always empty
                            //std::cout << "cluster = " << i << "\n";
                            for ( int j = 0; j < clusters[i].size();j ++){
                                //std::cout << "x = " << clusters.at(i).at(j).pt.x << "y = " << clusters.at(i).at(j).pt.y << "j = " << j << "\n";
                                avgX = avgX + clusters.at(i).at(j).pt.x;
                            }
                            avgX = avgX/double(clusters[i].size());
                            //std::cout << "avgX = " << avgX << "\n";
                            break;


                        }
                        if  (truthChar == 'P'){
                            correct += 1;
                        }
                        else{
                            error += 1;
                            falsePos += 1;
                        }
                    }
//                        for (int i = 0; i < clusters[1].size(); i ++){
//                            //std::vector<int>::iterator it;


//                            if (find (keypointsW.begin().pt.x, keypointsW.end().pt.x, clusters.at(1).at(i).pt.x) != keypointsW.end().pt.x){
//                             whiteCount != 1;

//                            }
//                            else if (find (keypointsB.begin().pt.x, keypointsB.end().pt.x, clusters.at(1).at(i).pt.x) != keypointsB.end().pt.x){
//                             blackCount != 1;

//                            }

//                        }


                    //}


                    else{
                        if (truthChar == 'P'){
                            error += 1;
                            falseNeg += 1;
                        }
                        else{
                            correct += 1;
                        }


                    }
                    //std::cout << "True Positive in white!!! correct count = " << correct << "\n";


					
                //}


                    //std::cout << "False Positive in white!!! error count = " << error << "\n";
                //}
                //Mat im_with_keypoints;
                Mat dst_with_keypoints;

                //drawKeypoints( im, keypointsW, im_with_keypoints, Scalar(0,255,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
                //drawKeypoints( im_with_keypoints, keypointsB, im_with_keypoints, Scalar(255,0,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
                drawKeypoints( im, keypointsW, dst_with_keypoints, Scalar(0,255,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
                drawKeypoints( dst_with_keypoints, keypointsB, dst_with_keypoints, Scalar(255,0,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
                // Show blobs
                //imshow("keypoints", im_with_keypoints );
                //imshow("keypointsAdaptiveThresh", dst_with_keypoints);
                imshow("BlackAndWhite", dst_with_keypoints );


            }

            else{
                if (truthChar == 'P'){
					error += 1;
                    falseNeg += 1;
                    //std::cout << "False Negative in white!!! error count = " << error << "\n";
                }
				else{
					correct += 1;
					//std::cout << "True Negative in white!!! correct count = " << correct << "\n";
				}
            }


            imshow( "Display window", im );

            //double percent = correct / (correct + error);
            //std::cout << "Percentage Correct is: " << percent << "\n";
            std::cout << "False Positive!!! count = " << falsePos << "\n";
            std::cout << "False Negative!!! count = " << falseNeg << "\n";
            double percent = correct / (correct + error);
            std::cout << "Percentage Correct is: " << percent << "\n";
            std::cout << "Correct is: " << correct << "\n";
            std::cout << "Error is: " << error << "\n";
            int keypress = waitKey(0); //saving the pressed key
            if (keypress == 27){ //if pressed Esc key
                im.release(); //turn off camera

                break;
            }

        }

    }
	t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Times passed in seconds: " << t << endl;

    std::cout << "False Positive!!! count = " << falsePos << "\n";
    std::cout << "False Negative!!! count = " << falseNeg << "\n";
    double percent = correct / (correct + error);
    std::cout << "Percentage Correct is: " << percent << "\n";
    std::cout << "Correct is: " << correct << "\n";
    std::cout << "Error is: " << error << "\n";
    closedir(directory);
    cv::destroyAllWindows();
}



/* DBSCAN - density-based spatial clustering of applications with noise */

void DBSCAN_keypoints(vector<KeyPoint> *keypoints, float eps, int minPts)
{
clusters.clear();;
vector<bool> clustered;
vector<int> noise;
vector<bool> visited;
vector<int> neighborPts;
vector<int> neighborPts_;
int c;

int noKeys = keypoints->size();

//init clustered and visited
for(int k = 0; k < noKeys; k++)
{
    clustered.push_back(false);
    visited.push_back(false);
}

//C =0;
c = 0;
clusters.push_back(vector<KeyPoint>()); //will stay empty?

//for each unvisted point P in dataset keypoints
for(int i = 0; i < noKeys; i++)
{
    if(!visited[i])
    {
        //Mark P as visited
        visited[i] = true;
        neighborPts = regionQuery(keypoints,&keypoints->at(i),eps);
        if(neighborPts.size() < minPts)
            //Mark P as Noise
            noise.push_back(i);
        else
        {
            clusters.push_back(vector<KeyPoint>());
            c++;
            //expand cluster
            // add P to cluster c
            clusters[c].push_back(keypoints->at(i));
            clustered[i] = true;
            //for each point P' in neighborPts
            for(int j = 0; j < neighborPts.size(); j++)
            {
                //if P' is not visited
                if(!visited[neighborPts[j]])
                {
                    //Mark P' as visited
                    visited[neighborPts[j]] = true;
                    neighborPts_ = regionQuery(keypoints,&keypoints->at(neighborPts[j]),eps);
                    if(neighborPts_.size() >= minPts)
                    {
                        neighborPts.insert(neighborPts.end(),neighborPts_.begin(),neighborPts_.end());
                    }
                }
                // if P' is not yet a member of any cluster
                // add P' to cluster c
                if(!clustered[neighborPts[j]])
                    clusters[c].push_back(keypoints->at(neighborPts[j]));
                    clustered[neighborPts[j]] = true;
            }
        }

    }
}
//return clusters;
//std::cout << "clusters count in dbscan = " << clusters.size() << "\n";
}

vector<int> regionQuery(vector<KeyPoint> *keypoints, KeyPoint *keypoint, float eps)
{
float dist;
vector<int> retKeys;
for(int i = 0; i< keypoints->size(); i++)
{
    dist = sqrt(pow((keypoint->pt.x - keypoints->at(i).pt.x),2)+pow((keypoint->pt.y - keypoints->at(i).pt.y),2));
    if(dist <= eps && dist != 0.0f)
    {
        retKeys.push_back(i);
    }
}
return retKeys;
}


