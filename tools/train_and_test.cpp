
#include <config.h>
#include <dataset.h>
#include <evaluation.h>
#include <classification_tree.h>
#include <differential_evolution.h>
#include <evolution_strategies.h>

#include <iostream>

using namespace evaboost;

namespace {

const std::string DIFFERENTIAL = "differential";
const std::string STRATEGIES = "strategies";

} // namespace

int main(int argc, const char** argv) {
    if (argc != 2) {
        throw std::runtime_error("Invalid paramaters");
    }

    std::cout << "Parse config file: " << std::string(argv[1]) << std::endl;
    Config config(argv[1]);

    std::cout << "Load train dataset: " << config.trainPath << std::endl;
    Dataset trainDataset = Dataset::loadFromCSV(config.trainPath, config.yPos);

    std::cout << "Load test dataset: " << config.testPath << std::endl;
    Dataset testDataset = Dataset::loadFromCSV(config.testPath, config.yPos);

    std::cout << "Train classification tree. Mode: " << config.mode;
    DifferentialEvolutionOptimizer optimizer(
        config.mutationFactor, config.crossoverProbability,
        config.populationSize, config.iterationsNumber,
        config.maxHeight, config.minItemInNode
    );
    ClassificationTree resultTree = [&](){
        if (DIFFERENTIAL == config.mode) {
            DifferentialEvolutionOptimizer optimizer(
                config.mutationFactor, config.crossoverProbability,
                config.populationSize, config.iterationsNumber,
                config.maxHeight, config.minItemInNode
            );
            return optimizer.train(trainDataset, config.cartPathes);
        } else if (STRATEGIES == config.mode) {
            EvolutionStrategiesOptimizer optimizer(
                config.samplesNumber, config.deviation,
                config.learningRate, config.iterationsNumber,
                config.maxHeight, config.minItemInNode
            );
            if (config.cartPathes.empty()) {
                return optimizer.train(trainDataset, "");
            } else {
                return optimizer.train(trainDataset, config.cartPathes.front());
            }
        } else {
            throw std::runtime_error("Invalid mode:" + config.mode);
        }
    }();

    double testAccuracy = accuracy(testDataset, resultTree);

    std::cout << "Accuracy: " << testAccuracy << std::endl;
    std::cout << "Done!" << std::endl;

    return 0;
}