// Copyright 2019 Dolotov Evgeniy

#ifndef EVABOOST_CONFIG_H
#define EVABOOST_CONFIG_H

#include <string>
#include <vector>

namespace evaboost {

class Config {
public:
    Config(const std::string& path);

    std::string testPath = "train.csv";
    std::string trainPath = "test.csv";
    size_t yPos = 0;
    std::string mode = "differential";
    double mutationFactor = 1.;
    double crossoverProbability = 0.8;
    size_t samplesNumber = 200;
    double deviation = 5.;
    double learningRate = 0.5;
    size_t populationSize = 100;
    size_t iterationsNumber = 1000;
    size_t maxHeight = 10;
    size_t minItemInNode = 3;
    std::vector<std::string> cartPathes;
};

} // namespace evaboost

#endif // EVABOOST_CONFIG_H