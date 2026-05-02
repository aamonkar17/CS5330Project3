/*
  Bruce A. Maxwell

  Set of utility functions for computing features and embeddings
*/
#include <cstdio>
#include <cstring>
#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#define _USE_MATH_DEFINES
#include <math.h>


/*
  cv::Mat src        thresholded and cleaned up image in 8UC3 format
  cv::Mat embedding  holds the embedding vector after the function returns
  cv::dnn::Net net   a pre-trained ResNet 18 network (modify the name of the output layer otherwise)
  int debug          1: show the image given to the network and print the embedding, 0: don't show extra info

 */

int getEmbedding( cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net, int debug ) {
  const int ORNet_size = 224; // expected network input size
  cv::Mat blob;
  cv::Mat resized;

  cv::resize( src, resized, cv::Size( ORNet_size, ORNet_size ) );
	
  cv::dnn::blobFromImage( resized, // input image
			  blob, // output array
			  (1.0/255.0) * (1/0.226), // scale factor
			  cv::Size( ORNet_size, ORNet_size ), // resize the image to this
			  cv::Scalar( 124, 116, 104),   // subtract mean prior to scaling
			  true, // swapRB
			  false,  // center crop after scaling short side to size
			  CV_32F ); // output depth/type

  net.setInput( blob );
  embedding = net.forward( "onnx_node!resnetv22_flatten0_reshape0" );

  if(debug) {
    std::cout << embedding << std::endl;
  }

  return(0);
}


/*
  Given the oriented bounding box information, extracts the region
  from the original image and rotates it so the primary axis is
  pointing right.
  
  cv::Mat &frame - the original image
  cv::Mat &embimage - the resulting ROI
  int cx - the x coordinate of the centroid of the region
  int cy - the y coordinate of the centroid of the region
  float theta - the orientation of the primary axis of the region (first eigenvector / least 2nd central moment
  float minE1 - the minimum projection value along the primary axis (should be a negative value)
  float maxE1 - the maximum projection value along the primary axis (should be a positive value)
  float minE2 - the minimum projection value along the secondary axis (should be a negative value)
  float maxE2 - the maximum projection value along the secondary axis (should be a positive value)
  int debug - whether to show intermediate images and values

  Note that the expectation is that maxE2 will correspond to up when
  maxE1 is pointing right.  If your system is computing maxE2 as
  pointing down, then swap minE2 and maxE2 and ensure that minE2 is
  negative and maxE2 is positive.
*/


void prepEmbeddingImage( cv::Mat &frame, cv::Mat &embimage, int cx, int cy, float theta, float minE1, float maxE1, float minE2, float maxE2, int debug ) {
    
    // rotate the image to align the primary region with the x-axis
    cv::Mat rotatedImage;
    cv::Mat M;

    M = cv::getRotationMatrix2D(
        cv::Point2f(cx, cy),
        -theta * 180.0 / M_PI,
        1.0
    );
    int largest = frame.cols > frame.rows ? frame.cols : frame.rows;
    largest = (int)(1.414 * largest);
    cv::warpAffine( frame, rotatedImage, M, cv::Size( largest, largest ) );

    if(debug) {
      cv::imshow( "rotated", rotatedImage );
    }

    int left = cx + (int)minE1;
    int top = cy - (int)maxE2;
    int width = (int)maxE1 - (int)minE1;
    int height = (int)maxE2 - (int)minE2;

    // bounds check the ROI
    if( left < 0 ) {
      width += left;
      left = 0;
    }
    if(top < 0 ) {
      height += top;
      top = 0;
    }
    if(left + width >= rotatedImage.cols) {
      width = (rotatedImage.cols-1) - left;
    }
    if(top + height >= rotatedImage.rows) {
      height = (rotatedImage.rows-1) - top;
    }

    if(debug) {
      printf("ROI box: %d %d %d %d\n", left, top, width, height );
    }

    // crop the image to the bounding box of the object
    cv::Rect  objroi( left, top, width, height);
    cv::rectangle( rotatedImage, cv::Point2d(objroi.x, objroi.y), cv::Point2d(objroi.x+objroi.width, objroi.y+objroi.height), 200, 4 );

    // extract the image and get the embedding of the original image
    cv::Mat extractedImage( rotatedImage, objroi );

    if(debug) {
      cv::imshow( "extracted", extractedImage );
    }
    
    extractedImage.copyTo(embimage);

    
    return;
}
