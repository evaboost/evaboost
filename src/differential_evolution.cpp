#include <dataset.h>
#include <evaluation.h>
#include <optimization.h>
#include <classification_tree.h>
#include <differential_evolution.h>

#include <cmath>
#include <algorithm>

namespace evaboost {

namespace {

std::vector<double> randomVector(size_t size, double minValue, double maxValue) {
    std::vector<double> result(size, 0);
    for (size_t i = 0; i < size; i++) {
        double r = rand() / double(RAND_MAX);
        result[i] = minValue + r * (maxValue - minValue);
    }
    return result;
}

} // namespace

DifferentialEvolutionOptimizer::DifferentialEvolutionOptimizer(
    double mutationFactor, double crossoverProbability,
    size_t populationSize, size_t iterationsNumber,
    size_t maxHeight, size_t minItemInNode)
    : mutationFactor_(mutationFactor),
      crossoverProbability_(crossoverProbability),
      populationSize_(populationSize),
      iterationsNumber_(iterationsNumber),
      maxHeight_(maxHeight),
      minItemInNode_(minItemInNode)
{}


ClassificationTree DifferentialEvolutionOptimizer::train(
    const Dataset& dataset, const std::vector<std::string>& cartPathes)
{
    std::vector<ClassificationTree> population;
    for (size_t i = 0; i < std::min(populationSize_, cartPathes.size()); i++) {
        population.push_back(ClassificationTree::fromCART(cartPathes[i]));
    }
    while (population.size() < populationSize_) {
        size_t vectorSize = pow(2, maxHeight_) - 1;
        std::vector<double> indexVector = randomVector(vectorSize, 0., 1.);
        std::vector<double> thresholdVector = randomVector(vectorSize, 0., 1.);
        population.push_back(
            ClassificationTree::fromRealVectors(
                indexVector, thresholdVector, dataset, minItemInNode_));
    }
    for (size_t iter = 0; iter < iterationsNumber_; iter++) {
        for (size_t i1 = 0; i1 < population.size(); i1++) {
            size_t i2 = rand() % population.size();
            size_t i3 = rand() % population.size();
            ClassificationTree mutatedTree
                = mutate(population[i1], population[i2], population[i3], dataset);
            population.push_back(crossover(population[i1], mutatedTree, dataset));
        }
        population = select(population, populationSize_, dataset);
    }
    return select(population, 1, dataset).front();
}

double DifferentialEvolutionOptimizer::mutationFactor() const {
    return mutationFactor_;
}

double DifferentialEvolutionOptimizer::crossoverProbability() const {
    return crossoverProbability_;
}

ClassificationTree DifferentialEvolutionOptimizer::mutate(
    const ClassificationTree& tree1,
    const ClassificationTree& tree2,
    const ClassificationTree& tree3,
    const Dataset& dataset) const
{
    std::vector<double> indexVector1 = tree1.indexVector();
    std::vector<double> thresholdVector1 = tree1.thresholdVector();
    std::vector<double> indexVector2 = tree2.indexVector();
    std::vector<double> thresholdVector2 = tree2.indexVector();
    std::vector<double> indexVector3 = tree3.indexVector();
    std::vector<double> thresholdVector3 = tree3.thresholdVector();

    std::vector<double> indexVector;
    std::vector<double> thresholdVector;
    for (size_t i = 0; i < indexVector1.size(); i++) {
        indexVector.push_back(
            indexVector1[i] + mutationFactor_ * (indexVector2[i] - indexVector3[i]));
    }
    for (size_t i = 0; i < thresholdVector1.size(); i++) {
        thresholdVector.push_back(
            thresholdVector1[i] + mutationFactor_ * (thresholdVector2[i] - thresholdVector3[i]));
    }
    return ClassificationTree::fromRealVectors(indexVector, thresholdVector, dataset, minItemInNode_);
}

ClassificationTree DifferentialEvolutionOptimizer::crossover(
    const ClassificationTree& tree1,
    const ClassificationTree& tree2,
    const Dataset& dataset) const
{
    std::vector<double> indexVector1 = tree1.indexVector();
    std::vector<double> thresholdVector1 = tree1.thresholdVector();
    std::vector<double> indexVector2 = tree2.indexVector();
    std::vector<double> thresholdVector2 = tree2.thresholdVector();

    std::vector<double> indexVector(indexVector1.size(), 0.);
    std::vector<double> thresholdVector(thresholdVector1.size(), 0.);

    size_t pos = rand() % indexVector1.size();

    for (size_t i = 0; i < indexVector1.size(); i++) {
        if (i != pos) {
            double p = rand() / double(RAND_MAX);
            if (p <= crossoverProbability_) {
                indexVector[i] = indexVector2[i];
            } else {
                indexVector[i] = indexVector1[i];
            }
        } else {
            indexVector[i] = indexVector2[i];
        }
    }

    for (size_t i = 0; i < thresholdVector1.size(); i++) {
        if (i != pos) {
            double p = rand() / double(RAND_MAX);
            if (p <= crossoverProbability_) {
                thresholdVector[i] = thresholdVector2[i];
            } else {
                thresholdVector[i] = thresholdVector1[i];
            }
        } else {
            indexVector[i] = thresholdVector2[i];
        }
    }
    return ClassificationTree::fromRealVectors(indexVector, thresholdVector, dataset, minItemInNode_);
}

std::vector<ClassificationTree> DifferentialEvolutionOptimizer::select(
    const std::vector<ClassificationTree>& population, size_t number,
    const Dataset& dataset) const
{
    std::vector<std::pair<ClassificationTree, double>> accuracies;
    for (size_t i = 0; i < population.size(); i++) {
        accuracies.push_back(
            std::make_pair(population[i], accuracy(dataset, population[i])));
    }
    std::sort(
        accuracies.begin(), accuracies.end(),
        [&](const std::pair<ClassificationTree, double>& tree1,
            const std::pair<ClassificationTree, double>& tree2)
        {
            return tree1.second > tree2.second;
        }
    );
    std::vector<ClassificationTree> selected;
    for (size_t i = 0; i < number; i++) {
        selected.push_back(accuracies[i].first);
    }
    return selected;
}

} // namespace evaboost