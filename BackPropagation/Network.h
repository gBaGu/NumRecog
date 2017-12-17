#pragma once
#include <vector>
#include <memory>
#include <thread>
#include <experimental\filesystem>

#include <opencv2\opencv.hpp>

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}
#include <LuaBridge.h>

#include "Neuron.h"

namespace fs = std::experimental::filesystem;


typedef std::vector<Neuron> Layer;

class Network
{
public:
	struct Config
	{
		fs::path trainSelectionPath;
		double learningRate;
		cv::Size inputImageSize;
		size_t hiddenLayerSize;
		size_t classes;
		fs::path loadWeightsPath;
		fs::path saveWeightsPath;
	};

	static std::unique_ptr<Network> load(lua_State* lua, const fs::path& scriptFilename);
	Network(Config cfg);

	void train();
	int detect(cv::Mat sample);

private:
	void loadWeights();
	void saveWeights() const;
	double doEpoch();
	double getTotalError(const std::vector<double>& targetOutput) const;
	void asyncShuffleTrainSelection();

	std::vector<std::pair<cv::Mat, std::string>> trainSelection;
	std::vector<std::pair<cv::Mat, std::string>> shuffledTrainSelection;
	std::unique_ptr<std::thread> shuffler;
	size_t epochNumber = 0;
	double learningRate = 0.5;
	fs::path loadWeightsPath;
	fs::path saveWeightsPath;
	cv::Size imageSize;

	Layer inputLayer;
	Bias inputBias;
	Layer hiddenLayer;
	Bias hiddenBias;
	Layer outputLayer;
};