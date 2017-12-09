#pragma once
#include <vector>

#include <opencv2\opencv.hpp>

#include "Neuron.h"


typedef std::vector<Neuron> Layer;

class Network
{
public:
	struct Config
	{
		std::string trainSelectionFolder;
		double learningRate;
		cv::Size inputImageSize;
		size_t hiddenLayerSize;
		size_t classes;
	};

	Network(Config cfg);

	void train();
	void test(cv::Mat sample);

private:
	double doEpoch();
	void passForward(cv::Mat sample);
	double getTotalError(const std::vector<double>& targetOutput) const;

	std::vector<std::pair<cv::Mat, std::string>> trainSelection;
	size_t epochNumber = 0;
	double learningRate = 0.5;

	Layer inputLayer;
	Layer hiddenLayer;
	Layer outputLayer;
};