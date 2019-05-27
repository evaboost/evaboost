// Copyright 2019 Dolotov Evgeniy

#include <dataset.h>
#include <evaluation.h>
#include <optimization.h>
#include <classification_tree.h>
#include <evolution_strategies.h>

#include <cmath>
#include <cstdlib>

namespace evaboost {

namespace {

std::vector<double> randomVector(size_t size, double minValue, double maxValue) {
    std::vector<double> result(size, 0.);
    for (size_t i = 0; i< size; i++) {
        double r = rand() / double(RAND_MAX);
        result[i] = minValue + r * (maxValue - minValue);
    }
    return result;
}

} // namespace

EvolutionStrategiesOptimizer::EvolutionStrategiesOptimizer(
    size_t samplesNumber, double deviation,
    double learningRate, size_t iterationsNumber,
    size_t maxHeight, size_t minItemInNode)
    : samplesNumber_(samplesNumber),
      deviation_(deviation),
      learningRate_(learningRate),
      iterationsNumber_(iterationsNumber),
      maxHeight_(maxHeight),
      minItemInNode_(minItemInNode)
{}

ClassificationTree EvolutionStrategiesOptimizer::train(
    const Dataset& dataset, const std::string& path) const
{
    ClassificationTree tree = [&](){
        if (!path.empty()) {
            return ClassificationTree::fromCART(path);
        } else {
             size_t vectorSize = pow(2, maxHeight_) - 1;
             std::vector<double> indexVector = randomVector(vectorSize, 0., 1.);
             std::vector<double> thresholdVector = randomVector(vectorSize, 0., 1.);
             return ClassificationTree::fromRealVectors(
                indexVector, thresholdVector, dataset, minItemInNode_);
        }
    }();
    for (size_t iter = 0; iter < iterationsNumber_; iter++) {
        std::vector<double> indexVector = tree.indexVector();
        std::vector<double> thresholdVector = tree.thresholdVector();
        double scale = learningRate_ / (samplesNumber_ * deviation_);
        for (size_t i = 0; i < samplesNumber_; i++) {
            std::vector<double> indexEpsilon;
            std::vector<double> thresholdEpsilon;
            ClassificationTree newTree
                = sampleTree(tree, dataset, indexEpsilon, thresholdEpsilon);
            std::vector<double> newIndexVector = newTree.indexVector();
            std::vector<double> newThresholdVector = newTree.thresholdVector();
            double value = accuracy(dataset, newTree);
            for (size_t j = 0; j < indexVector.size(); i++) {
                indexVector[j] += scale * value * indexEpsilon[j];
            }
            for (size_t j = 0; j < indexVector.size(); i++) {
                indexVector[j] += scale * value * thresholdEpsilon[j];
            }
        }
        tree = ClassificationTree::fromRealVectors(
                   indexVector, thresholdVector, dataset, minItemInNode_);
    }
    return tree;
}

ClassificationTree EvolutionStrategiesOptimizer::sampleTree(
    const ClassificationTree& tree, const Dataset& dataset,
    std::vector<double>& indexEpsilon, std::vector<double>& thresholdEpsilon) const
{
    std::vector<double> indexVector = tree.indexVector();
    std::vector<double> thresholdVector = tree.thresholdVector();

    for (size_t i = 0; i < indexVector.size(); i++) {
        double epsilon = 2 * (rand() / double(RAND_MAX)) - 1;
        indexEpsilon.push_back(epsilon);
        indexVector[i] += deviation_ * epsilon;
    }
    for (size_t i = 0; i < thresholdVector.size(); i++) {
        double epsilon = 2 * (rand() / double(RAND_MAX)) - 1;
        thresholdEpsilon.push_back(epsilon);
        thresholdVector[i] += deviation_ * epsilon;
    }
    return ClassificationTree::fromRealVectors(
        indexVector, thresholdVector, dataset, minItemInNode_);
}

} // namespace evaboost