#include <dataset.h>

#include <fstream>
#include <exception>
#include <stdexcept>
#include <sstream>
#include <cstdlib>

namespace evaboost {

namespace {

std::vector<std::string> splitByComma(const std::string& str) {
    std::vector<std::string> items;
    std::stringstream ss(str);
    while(!ss.eof()) {
        std::string item;
        std::getline(ss, item, ',');
        if (item.empty()) {
            continue;
        }
        items.push_back(item);
    }
    return items;
}

size_t getPos(const std::string& paramsStr, const std::string& name) {
    std::vector<std::string> items = splitByComma(paramsStr);
    for (size_t pos = 0; pos < items.size(); pos++) {
        if (items[pos] == name) {
            return pos;
        }
    }
    throw std::runtime_error("Failed to find position for name:" + name);
}

DataItem parseItem(const std::string& itemStr, size_t yPos) {
    std::vector<std::string> items = splitByComma(itemStr);
    int y = std::stoi(items[yPos]);
    std::vector<double> features;
    for (size_t pos = 0; pos < items.size(); pos++) {
        if (pos != yPos) {
            y = std::stod(items[pos]);
        }
    }
    return DataItem(features, y);
}

} // namespace


Dataset Dataset::loadFromCSV(const std::string& path, size_t yPos) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open file:" + path);
    }
    Dataset dataset;
    while (!ifs.eof()) {
        std::string itemStr;
        std::getline(ifs, itemStr);
        if (itemStr.empty()) {
            continue;
        }
        dataset.addItem(parseItem(itemStr, yPos));
    }
    return dataset;
}

Dataset Dataset::loadFromCSV(const std::string& path, const std::string& yName) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::string headerStr;
    std::getline(ifs, headerStr);
    size_t yPos = getPos(headerStr, yName);
    Dataset dataset;
    while (!ifs.eof()) {
        std::string itemStr;
        std::getline(ifs, itemStr);
        if (itemStr.empty()) {
            continue;
        }
        dataset.addItem(parseItem(itemStr, yPos));
    }
    return dataset;
}

Dataset::iterator Dataset::begin() {
    return items_.begin();
}

Dataset::const_iterator Dataset::begin() const {
    return items_.cbegin();
}

Dataset::const_iterator Dataset::cbegin() const {
    return items_.cbegin();
}

Dataset::iterator Dataset::end() {
    return items_.end();
}

Dataset::const_iterator Dataset::end() const {
    return items_.cend();
}

Dataset::const_iterator Dataset::cend() const {
    return items_.cend();
}

size_t Dataset::itemsNumber() const {
    return items_.size();
}

DataItem Dataset::itemAt(size_t i) const {
    if (i < items_.size()) {
        return items_[i];
    } else {
        throw std::runtime_error("Invalid item index:" + std::to_string(i));
    }
}

size_t Dataset::featuresNumber() const {
    if (!items_.empty()) {
        return items_.front().featuresNumber();
    } else {
        throw std::runtime_error("Dataset is empty");
    }
}

size_t Dataset::classesNumber() const {
    return classes_.size();
}

void Dataset::addItem(const DataItem& item) {
    if (items_.empty()) {
        items_.push_back(item);
        classes_.insert(item.y());
    } else if (items_.back().featuresNumber() == item.featuresNumber()) {
        items_.push_back(item);
        classes_.insert(item.y());
    } else {
        throw std::runtime_error(
            "Invalid features number: " + std::to_string(item.featuresNumber()));
    }
}

Dataset Dataset::sampleDataset(size_t datasetSize) const {
    Dataset dataset;
    dataset.classes_ = classes_;
    for (size_t i = 0; i < datasetSize; i++) {
        size_t pos = rand() % items_.size();
        dataset.addItem(items_[pos]);
    }
    return dataset;
}

} // namespace evaboost