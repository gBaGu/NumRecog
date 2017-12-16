#include <fstream>
#include <thread>
#include <algorithm>
#include <iterator>
#include <experimental\filesystem>

#include "Network.h"

namespace fs = std::experimental::filesystem;


Network::Network(Config cfg)
	: learningRate(cfg.learningRate)
{
	for (auto& p : fs::directory_iterator(cfg.trainSelectionFolder))
	{
		if (p.path().extension().string() != ".jpg")
		{
			continue;
		}
		std::ifstream inf(p.path().parent_path().string() + "/" + p.path().stem().string() + ".txt");
		std::string label;
		inf >> label;

		auto img = cv::imread(p.path().string(), CV_LOAD_IMAGE_GRAYSCALE);
		cv::resize(img, img, cfg.inputImageSize);
		trainSelection.push_back(std::pair<cv::Mat, std::string>(img, label));
	}
	std::transform(trainSelection.begin(), trainSelection.end(), std::back_inserter(shuffledTrainSelection),
		[](std::pair<cv::Mat, std::string> p)
	{
		return std::pair<cv::Mat, std::string>(p.first.clone(), p.second);
	});

	inputLayer.resize(cfg.inputImageSize.area());
	hiddenLayer.resize(cfg.hiddenLayerSize);
	outputLayer.resize(cfg.classes);

	for (auto& lIn : inputLayer)
	{
		for (auto& lOut : hiddenLayer)
		{
			lIn.addLink(&lOut);
		}
	}
	for (auto& lOut : hiddenLayer)
	{
		inputBias.addLink(&lOut);
	}
	for (auto& lIn : hiddenLayer)
	{
		for (auto& lOut : outputLayer)
		{
			lIn.addLink(&lOut);
		}
	}
	for (auto& lOut : outputLayer)
	{
		hiddenBias.addLink(&lOut);
	}
}

void Network::train()
{
	double error = 0;
	do
	{
		try
		{
			error = doEpoch();
		}
		catch (const std::exception& ex)
		{
			std::cout << ex.what() << std::endl;
			exit(1);
		}
		std::cout << "\n==============================================\n"
			<< "EPOCH ENDED\n"
			<< "Epoch: " << epochNumber << "\tError: " << error << std::endl
			<< "==============================================\n";
		learningRate *= 0.99;
	} while (error > 0.01 && epochNumber < 200);
}

double Network::doEpoch()
{
	asyncShuffleTrainSelection();

	epochNumber++;
	size_t mainPartitionSize = trainSelection.size() * 0.7;
	auto partition = trainSelection.begin() + mainPartitionSize;
	//BACK PROPAGATION
	for (auto sampleIt = trainSelection.begin(); sampleIt != partition; sampleIt++)
	{
		std::cout << "Winner: " << detect(sampleIt->first);
		std::cout << "\t\tRight answer: " << sampleIt->second << std::endl;

		std::vector<double> targetOutput(outputLayer.size(), 0.0);
		targetOutput.at(stoi(sampleIt->second)) = 1.0;

		//CALCULATING HIDDEN->OUTPUT
		auto targetIt = targetOutput.begin();
		for (auto neuron : outputLayer)
		{
			for (auto inLink : neuron.getInputLinks())
			{
				auto outSignal = neuron.getOutputSignal();
				double firstPart = -(*targetIt - outSignal);
				double secondPart = outSignal * (1 - outSignal);
				double thirdPart = inLink->getInputNeuron()->getOutputSignal();

				double gradientWithRespectToWeight = firstPart * secondPart * thirdPart;
				inLink->calculateNewWeight(gradientWithRespectToWeight, learningRate);
			}
			targetIt++;
		}
		//CALCULATING INPUT->HIDDEN
		for (auto neuron : hiddenLayer)
		{
			for (auto inLink : neuron.getInputLinks())
			{
				auto outSignal = neuron.getOutputSignal();
				double firstPart = 0;
				targetIt = targetOutput.begin();
				for (auto outLink : neuron.getOutputLinks())
				{
					auto outSignal = outLink->getOutputNeuron()->getOutputSignal();
					firstPart += (-(*targetIt - outSignal) * outSignal * (1 - outSignal) * outLink->getWeight());
					targetIt++;
				}
				double secondPart = outSignal * (1 - outSignal);
				double thirdPart = inLink->getInputNeuron()->getOutputSignal();

				double gradientWithRespectToWeight = firstPart * secondPart * thirdPart;
				inLink->calculateNewWeight(gradientWithRespectToWeight, learningRate);
			}
		}
		//APPLYING INPUT->HIDDEN
		for (auto neuron : inputLayer)
		{
			for (auto outLink : neuron.getOutputLinks())
			{
				outLink->applyNewWeight();
			}
		}
		//APPLYING HIDDEN->OUTPUT
		for (auto neuron : hiddenLayer)
		{
			for (auto outLink : neuron.getOutputLinks())
			{
				outLink->applyNewWeight();
			}
		}
	}
	//CALCULATING TOTAL ERROR
	double totalError = 0.0;
	for (auto sampleIt = partition; sampleIt != trainSelection.end(); sampleIt++)
	{
		std::cout << "Winner: " << detect(sampleIt->first);
		std::cout << "\t\tRight answer: " << sampleIt->second << std::endl << std::endl;

		std::vector<double> targetOutput(outputLayer.size(), 0.0);
		targetOutput.at(stoi(sampleIt->second)) = 1.0;

		totalError += getTotalError(targetOutput);
	}

	shuffler->join();
	std::swap(shuffledTrainSelection, trainSelection);

	return totalError / (trainSelection.size() - mainPartitionSize);
}

int Network::detect(cv::Mat sample)
{
	auto layerIt = inputLayer.begin();
	for (int i = 0; i < sample.rows; i++)
	{
		for (int j = 0; j < sample.cols; j++)
		{
			double signal = sample.at<uchar>(i, j) / 255.0;
			layerIt->translateSignal(signal);
			layerIt++;
		}
	}
	inputBias.translateSignal();

	for (layerIt = hiddenLayer.begin(); layerIt != hiddenLayer.end(); layerIt++)
	{
		layerIt->translateSignal();
	}
	hiddenBias.translateSignal();

	auto winner = std::max_element(outputLayer.begin(), outputLayer.end(), [](Neuron left, Neuron right)
	{
		return left.getOutputSignal() < right.getOutputSignal();
	});
	/*int i = 0;
	for (auto neuron : outputLayer)
	{
		std::cout << i << ": " << neuron.getOutputSignal() << std::endl;
		i++;
	}*/
	return std::distance(outputLayer.begin(), winner);
}

double Network::getTotalError(const std::vector<double>& targetOutput) const
{
	double res = 0.0;
	for (int i = 0; i < targetOutput.size(); i++)
	{
		res += 0.5 * pow(targetOutput[i] - outputLayer[i].getOutputSignal(), 2);
	}
	return res;
}

void Network::asyncShuffleTrainSelection()
{
	shuffler = std::make_unique<std::thread>([](std::vector<std::pair<cv::Mat, std::string>>& toShuffle)
	{
		std::random_shuffle(toShuffle.begin(), toShuffle.end());
	},
		shuffledTrainSelection);
}