#include <iostream>
#include <fstream>
#include <ctime>
#include <memory>
#include <experimental\filesystem>

#include <opencv2\opencv.hpp>

#include "Network.h"

namespace fs = std::experimental::filesystem;
//using namespace luabridge;


int main(int argc, char ** argv)
{
	srand(time(0));
	lua_State* lua = luaL_newstate();
	luaL_openlibs(lua);

	std::unique_ptr<Network> net;
	try
	{
		net = Network::load(lua, "data/config.lua");
	}
	catch (const std::runtime_error& ex)
	{
		std::cout << ex.what() << std::endl;
		return 0;
	}

	std::string testDir;
	if (argc == 3 && !strcmp(argv[1], "train"))
	{
		net->train();
		testDir = argv[2];
	}
	else if (argc == 2)
	{
		testDir = argv[1];
	}

	std::cout << "\n==============================================\n"
		<< "TESTING THIS SUPER DEEP (1 HIDDEN LAYER) NEURAL NETWORK\n"
		<< "==============================================\n";
	size_t testSelectionSize = 0;
	size_t rightAnswers = 0;
	size_t wrongAnswers = 0;
	for (auto& de : fs::directory_iterator(testDir))
	{
		auto path = de.path();
		if (path.extension().string() != ".jpg")
		{
			continue;
		}

		auto img = cv::imread(path.string(), CV_LOAD_IMAGE_GRAYSCALE);
		auto dtectedNum = net->detect(img);

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