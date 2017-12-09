#include <iostream>
#include <ctime>

#include <opencv2\opencv.hpp>

#include "Network.h"


int main(int argc, char ** argv)
{
	srand(time(0));

	Network::Config cfg;
	cfg.trainSelectionFolder = "img";
	cfg.learningRate = 0.5;
	cfg.inputImageSize = cv::Size(25, 50);
	cfg.hiddenLayerSize = 20;
	cfg.classes = 3;

	Network nn(cfg);
	nn.train();

	auto img = cv::imread("test/2.jpg");
	cv::resize(img, img, cfg.inputImageSize);
	nn.test(img);

	img = cv::imread("test/1.jpg");
	cv::resize(img, img, cfg.inputImageSize);
	nn.test(img);

	getchar();
	return 0;
}