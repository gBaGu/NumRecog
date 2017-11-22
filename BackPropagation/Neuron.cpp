#define _USE_MATH_DEFINES

#include <cmath>
#include <ctime>
#include <algorithm>

#include "Neuron.h"


double activationF(double val)
{
	return 1.0 / (1 + pow(M_E, -val));
}


NeuralLink::NeuralLink(Neuron* const inN, Neuron* const outN)
	: inputNeuron(inN), outputNeuron(outN)
{
	srand(time(0));
	weight = (rand() % 101) / 100 - 0.5;
}


void Neuron::addLink(Neuron* const n)
{
	std::shared_ptr<NeuralLink> link = std::make_shared<NeuralLink>(this, n);
	addOutputLink(link);
	n->addInputLink(link);
}

void Neuron::translateSignal() const
{
	auto outSignal = getOutputSignal();
	std::for_each(outputLinks.begin(), outputLinks.end(), [outSignal](std::shared_ptr<NeuralLink> link)
	{
		link->passSignal(outSignal);
	});
}

double Neuron::getOutputSignal() const
{
	auto sum = getSignalsSum();
	double outSignal = 0;
	outSignal = activationF(sum);
	return outSignal;
}

void Neuron::addInputLink(std::shared_ptr<NeuralLink> link)
{
	inputLinks.push_back(link);
}

void Neuron::addOutputLink(std::shared_ptr<NeuralLink> link)
{
	outputLinks.push_back(link);
}

double Neuron::getSignalsSum() const
{
	double sum = 0.0;
	std::for_each(inputLinks.begin(), inputLinks.end(), [&sum](std::shared_ptr<NeuralLink> link)
	{
		sum += link->getWeightedSignal();
	});
	return sum;
}