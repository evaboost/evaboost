#include <config.h>

#include <string>
#include <sstream>
#include <fstream>
#include <exception>
#include <stdexcept>

namespace evaboost {

namespace {

const std::string TEST_PATH_PARAM = "test_path";
const std::string TRAIN_PATH_PARAM = "train_path";
const std::string Y_POS_PARAM = "y_pos";
const std::string MODE_PARAM = "mode";
const std::string MUTATION_FACTOR_PARAM = "mutation_factor";
const std::string CROSSOVER_PROBABILITY_PARAM = "crossover_probability";
const std::string SAMPLES_NUMBER_PARAM = "samples_number";
const std::string DEVIATION_PARAM = "deviation";
const std::string LEARNING_RATE_PARAM = "learning_rate";
const std::string POPULATION_SIZE_PARAM = "population_size";
const std::string ITERATIONS_NUMBER_PARAM = "iterations_number";
const std::string MAX_HEIGHT_PARAM = "max_height";
const std::string MIN_ITEM_IN_NODE_PARAM = "min_item_in_node";
const std::string CART_PATHES_PARAM = "cart_pathes";

std::vector<std::string> splitByComma(const std::string& str) {
    std::vector<std::string> items;
    std::stringstream ss(str);
    while (!ss.eof()) {
        std::string item;
        std::getline(ss, item, ',');
        if (item.empty()) {
            continue;
        }
        items.push_back(item);
    }
    return items;
}

} // namespace

Config::Config(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    while (!ifs.eof()) {
        std::string type;
        ifs >> type;
        if (type.empty()) {
            continue;
        }
        if (TEST_PATH_PARAM == type) {
            ifs >> testPath;
        } else if (TRAIN_PATH_PARAM == type) {
            ifs >> trainPath;
        } else if (CART_PATHES_PARAM == type) {
            std::string str;
            ifs >> str;
            cartPathes = splitByComma(str);
        } else if (Y_POS_PARAM == type) {
            ifs >> yPos;
        } else if (MODE_PARAM == type) {
            ifs >> mode;
        } else if (MUTATION_FACTOR_PARAM == type) {
            ifs >> mutationFactor;
        } else if (CROSSOVER_PROBABILITY_PARAM == type) {
            ifs >> crossoverProbability;
        } else if (SAMPLES_NUMBER_PARAM == type) {
            ifs >> samplesNumber;
        } else if (DEVIATION_PARAM == type) {
            ifs >> deviation;
        } else if (LEARNING_RATE_PARAM == type) {
            ifs >> learningRate;
        } else if (POPULATION_SIZE_PARAM == type) {
            ifs >> populationSize;
        } else if (ITERATIONS_NUMBER_PARAM == type) {
            ifs >> iterationsNumber;
        } else if (MAX_HEIGHT_PARAM == type) {
            ifs >> maxHeight;
        } else if (MIN_ITEM_IN_NODE_PARAM == type) {
            ifs >> minItemInNode;
        } else {
            throw std::runtime_error("Invalid param: " + type);
        }
    }
}

} // namespace evabost