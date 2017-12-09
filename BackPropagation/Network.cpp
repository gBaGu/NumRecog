#include <fstream>
#include <thread>
#include <algorithm>
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
	for (auto& lIn : hiddenLayer)
	{
		for (auto& lOut : outputLayer)
		{
			lIn.addLink(&lOut);
		}
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
		}
		std::cout << "\n==============================================\n"
			<< "EPOCH ENDED\n"
			<< "Epoch: " << epochNumber << "\tError: " << error << std::endl
			<< "==============================================\n";
		learningRate *= 0.9;
	} while (error > 0.01 && epochNumber < 20);
}

void Network::test(cv::Mat sample)
{
	std::cout << "\n==============================================\n"
		<< "TESTING THIS SUPER DEEP (1 HIDDEN LAYER) NEURAL NETWORK\n"
		<< "==============================================\n";
	passForward(sample);
}

double Network::doEpoch()
{
	std::random_shuffle(trainSelection.begin(), trainSelection.end());

	epochNumber++;
	double totalError = 0.0;
	for (auto& sample : trainSelection)
	{
		passForward(sample.first);
		std::cout << "Right answer: " << sample.second << std::endl;

		std::vector<double> targetOutput(outputLayer.size(), 0.0);
		targetOutput.at(stoi(sample.second)) = 1.0;

		totalError += getTotalError(targetOutput);

		//CALCULATING HIDDEN->OUTPUT
		auto targetIt = targetOutput.begin();
		for (auto neuron : outputLayer)
		{
			for (auto inLink : neuron.getInputLinks())
			{
				auto outSignal = neuron.getOutputSignal();
				double firstPart = -(*targetIt - outSignal);
				//std::cout << "First part: " << firstPart << "\t";
				double secondPart = outSignal * (1 - outSignal);
				//std::cout << "Second part: " << secondPart << "\t";
				double thirdPart = inLink->getInputNeuron()->getOutputSignal();
				//std::cout << "Third part: " << thirdPart << "\t";

				double gradientWithRespectToWeight = firstPart * secondPart * thirdPart;
				//std::cout << "Gradient: " << gradientWithRespectToWeight << std::endl;
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
				//std::cout << outLink->getWeight() << "\t->\t";
				outLink->applyNewWeight();
				//std::cout << outLink->getWeight() << std::endl;
			}
		}
	}

	return totalError / trainSelection.size();
}

void Network::passForward(cv::Mat sample)
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

	for (layerIt = hiddenLayer.begin(); layerIt != hiddenLayer.end(); layerIt++)
	{
		layerIt->translateSignal();
	}

	auto winner = std::max_element(outputLayer.begin(), outputLayer.end(), [](Neuron left, Neuron right)
	{
		return left.getOutputSignal() < right.getOutputSignal();
	});
	int i = 0;
	for (auto neuron : outputLayer)
	{
		std::cout << i << ": " << neuron.getOutputSignal() << std::endl;
		i++;
	}
	std::cout << "Winner: " << std::distance(outputLayer.begin(), winner) << std::endl;
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