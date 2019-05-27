// Copyright 2019 Dolotov Evgeniy

#ifndef EVABOOST_DIFFERENTIAL_EVOLUTION_H
#define EVABOOST_DIFFERENTIAL_EVOLUTION_H

#include <dataset.h>
#include <optimization.h>
#include <classification_tree.h>

#include <vector>
#include <string>

namespace evaboost {

class DifferentialEvolutionOptimizer {
public:
    DifferentialEvolutionOptimizer(
        double mutationFactor, double crossoverProbability,
        size_t populationSize, size_t iterationsNumber,
        size_t maxHeight, size_t minItemInNode);

    ClassificationTree train(
        const Dataset& dataset, const std::vector<std::string>& cartPathes);

    double mutationFactor() const;

    double crossoverProbability() const;

private:
    ClassificationTree mutate(
        const ClassificationTree& tree1,
        const ClassificationTree& tree2,
        const ClassificationTree& tree3,
        const Dataset& dataset) const;

    ClassificationTree crossover(
        const ClassificationTree& tree1,
        const ClassificationTree& tree2,
        const Dataset& dataset) const;

    std::vector<ClassificationTree> select(
        const std::vector<ClassificationTree>& population, size_t number,
        const Dataset& dataset) const;

    double mutationFactor_;
    double crossoverProbability_;
    size_t populationSize_;
    size_t iterationsNumber_;
    size_t maxHeight_;
    size_t minItemInNode_;
};

} // namespace evaboost

#endif // EVABOOST_DIFFERENTIAL_EVOLUTION_H