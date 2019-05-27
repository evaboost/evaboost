// Copyright 2019 Dolotov Evgeniy

#ifndef EVABOOST_EVOLUTION_STRATEGIES_H
#define EVABOOST_EVOLUTION_STRATEGIES_H

#include <dataset.h>
#include <optimization.h>
#include <classification_tree.h>

#include <string>

namespace evaboost {

class EvolutionStrategiesOptimizer {
public:
    EvolutionStrategiesOptimizer(
        size_t samplesNumber, double deviation,
        double learningRate, size_t iterationsNumber,
        size_t maxHeight, size_t minItemInNode);

    ClassificationTree train(
        const Dataset& dataset, const std::string& path) const;

private:
    ClassificationTree sampleTree(
        const ClassificationTree& tree, const Dataset& dataset,
        std::vector<double>& indexEpsilon, std::vector<double>& thresholdEpsilon) const;

    size_t samplesNumber_;
    double deviation_;
    double learningRate_;
    size_t iterationsNumber_;
    size_t maxHeight_;
    size_t minItemInNode_;
};

} // namespace evaboost

#endif // EVABOOST_EVOLUTION_STRATEGIES_H