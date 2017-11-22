#pragma once
#include <vector>
#include <memory>


double activationF(double val);

class Neuron;
class NeuralLink
{
public:
	NeuralLink(Neuron* const inN, Neuron* const outN);

	void passSignal(double s) { signal = s; }
	double getWeightedSignal() const { return signal * weight; }

private:
	double weight;
	double signal;

	Neuron* inputNeuron;
	Neuron* outputNeuron;
};

class Neuron
{
public:
	Neuron() {}

	void addLink(Neuron* const n);
	void translateSignal() const;
	double getOutputSignal() const;

private:
	void addInputLink(std::shared_ptr<NeuralLink> link);
	void addOutputLink(std::shared_ptr<NeuralLink> link);
	double getSignalsSum() const;

	std::vector<std::shared_ptr<NeuralLink>> inputLinks;
	std::vector<std::shared_ptr<NeuralLink>> outputLinks;
};