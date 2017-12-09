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
	void calculateNewWeight(double gradient, double learningRate);
	void applyNewWeight() { weight = newWeight; }
	double getWeightedSignal() const { return signal * weight; }
	double getWeight() const { return weight; }
	Neuron* getInputNeuron() const { return inputNeuron; }
	Neuron* getOutputNeuron() const { return outputNeuron; }

private:
	double weight;
	double newWeight;
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
	void translateSignal(double s) const;
	double getOutputSignal() const;
	std::vector<std::shared_ptr<NeuralLink>> getInputLinks() const { return inputLinks; }
	std::vector<std::shared_ptr<NeuralLink>> getOutputLinks() const { return outputLinks; }

private:
	void addInputLink(std::shared_ptr<NeuralLink> link);
	void addOutputLink(std::shared_ptr<NeuralLink> link);
	double getSignalsSum() const;

	std::vector<std::shared_ptr<NeuralLink>> inputLinks;
	std::vector<std::shared_ptr<NeuralLink>> outputLinks;
};