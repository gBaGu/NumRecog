#pragma once
#include <vector>
#include <memory>
#include <thread>

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
	int detect(cv::Mat sample);

private:
	double doEpoch();
	double getTotalError(const std::vector<double>& targetOutput) const;

	void asyncShuffleTrainSelection();

	std::vector<std::pair<cv::Mat, std::string>> trainSelection;
	std::vector<std::pair<cv::Mat, std::string>> shuffledTrainSelection;
	std::unique_ptr<std::thread> shuffler;
	size_t epochNumber = 0;
	double learningRate = 0.5;

	Layer inputLayer;
	Bias inputBias;
	Layer hiddenLayer;
	Bias hiddenBias;
	Layer outputLayer;
};