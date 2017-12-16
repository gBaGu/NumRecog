#include <iostream>
#include <fstream>
#include <ctime>
#include <experimental\filesystem>

#include <opencv2\opencv.hpp>

#include "Network.h"

namespace fs = std::experimental::filesystem;


int main(int argc, char ** argv)
{
	srand(time(0));

	Network::Config cfg;
	cfg.trainSelectionFolder = "data/train_5x7_10";
	cfg.learningRate = 0.5;
	cfg.inputImageSize = cv::Size(5, 7);
	cfg.hiddenLayerSize = 50;//20;
	cfg.classes = 10;

	Network nn(cfg);
	nn.train();

	std::cout << "\n==============================================\n"
		<< "TESTING THIS SUPER DEEP (1 HIDDEN LAYER) NEURAL NETWORK\n"
		<< "==============================================\n";
	size_t testSelectionSize = 0;
	size_t rightAnswers = 0;
	size_t wrongAnswers = 0;
	for (auto& de : fs::directory_iterator("data/test_5x7_10"))
	{
		auto path = de.path();
		if (path.extension().string() != ".jpg")
		{
			continue;
		}

		auto img = cv::imread(path.string(), CV_LOAD_IMAGE_GRAYSCALE);
		cv::resize(img, img, cfg.inputImageSize);
		auto dtectedNum = nn.detect(img);

		testSelectionSize++;
		std::ifstream inf(path.parent_path() / path.filename().replace_extension(".txt"));
		if (inf.is_open())
		{
			int rightAnswer = -1;
			inf >> rightAnswer;
			if (dtectedNum == rightAnswer)
			{
				rightAnswers++;
			}
			else
			{
				wrongAnswers++;
			}
		}
	}
	std::cout << "RESULTS:\n"
		<< "Test selection size: " << testSelectionSize << std::endl
		<< "Right answers: " << rightAnswers << std::endl
		<< "Wrong answers: " << wrongAnswers << std::endl
		<< "Error: " << static_cast<double>(wrongAnswers) * 100.0 / static_cast<double>(testSelectionSize) << "%\n";

	getchar();
	return 0;
}