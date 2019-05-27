#include <node.h>
#include <classification_tree.h>

#include <cmath>
#include <queue>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <unordered_map>

namespace evaboost {

namespace {

int getClass(const std::vector<DataItem>& items) {
    std::unordered_map<int, size_t> counts;
    for (const DataItem& item : items) {
        counts[item.y()]++;
    }
    size_t maxCount = counts.begin()->second;
    int maxClass = counts.begin()->first;
    for (const std::pair<int, size_t>& classAndCount : counts) {
        if (classAndCount.second > maxCount) {
            maxCount = classAndCount.second;
            maxClass = classAndCount.first;
        }
    }
    return maxClass;
}

} // namespace

void ClassificationTree::toRealVectors(
    std::vector<double>& indexVector, std::vector<double>& thresholdVector) const
{
    indexVector.resize(featuresNumber(), -1.);
    thresholdVector.resize(featuresNumber(), -1.);
    size_t height = log(nodes_.size() + 1);
    size_t n = nodes_.size() - pow(2, height - 1);
    for (size_t i = 0; i < n; i++) {
        if (nodes_[i].isNull() || nodes_[i].isLeaf()) {
            bool needResize = true;
            size_t vectorSize = indexVector.size();
            for (size_t j = 0; j < indexVector.size(); j++) {
                if (indexVector[j] == -1.) {
                    needResize = false;
                    indexVector[j] = i / double(n);
                    thresholdVector[j] = 1.;
                    break;
                }
            }
            if (needResize) {
                indexVector.resize(vectorSize + featuresNumber(), -1.);
                thresholdVector.resize(vectorSize + featuresNumber(), -1.);
                indexVector[vectorSize] = i / double(n);
                thresholdVector[vectorSize] = 1.;
            }
        } else {
            bool needResize = true;
            size_t vectorSize = indexVector.size();
            for (size_t j = nodes_[i].featureIndex(); j < indexVector.size(); j += featuresNumber()) {
                if (indexVector[j] == -1.) {
                    needResize = false;
                    indexVector[j] = i / double(n);
                    thresholdVector[j] = nodes_[i].threshold();
                    break;
                }
            }
            if (needResize) {
                indexVector.resize(vectorSize + featuresNumber(), -1.);
                thresholdVector.resize(vectorSize + featuresNumber(), -1.);
                indexVector[vectorSize + nodes_[i].featureIndex()] = i / double(n);
                thresholdVector[vectorSize + nodes_[i].featureIndex()] = nodes_[i].threshold();
            }
        }
    }
}

ClassificationTree ClassificationTree::fromCART(const std::string& path) {
    ClassificationTree tree;
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::string type;
    ifs >> type;
    ifs >> tree.featuresNumber_;
    while (!ifs.eof()) {
        ifs >> type;
        if ("null" == type) {
            tree.nodes_.push_back(Node());
        } else if ("leaf" == type) {
            int y;
            ifs >> y;
            tree.nodes_.push_back(Node(y));
        } else if ("node" == type) {
            size_t featureIndex;
            double threshold;
            ifs >> featureIndex >> threshold;
            tree.nodes_.push_back(Node(featureIndex, threshold));
        } else {
            throw std::runtime_error("Invalid node type: " + type);
        }
    }
    tree.toRealVectors(tree.indexVector_, tree.thresholdVector_);
    return tree;
}

ClassificationTree ClassificationTree::fromRealVectors(
    const std::vector<double>& indexVector,
    const std::vector<double>& thresholdVector,
    const Dataset& dataset,
    size_t minItemInLeaf)
{
    ClassificationTree tree;
    tree.indexVector_ = indexVector;
    tree.thresholdVector_ = thresholdVector;

    tree.featuresNumber_ = dataset.featuresNumber();

    std::vector<std::pair<size_t, double>> indices;
    for (size_t i = 0; i < indexVector.size(); i++) {
        indices.push_back(std::make_pair(i, indexVector[i]));
    }
    std::sort(
        indices.begin(), indices.end(),
        [&](const std::pair<size_t, double>& lhs,
            const std::pair<size_t, double>& rhs) {
            return lhs.second < rhs.second;
        }
    );
    double maxValue = std::numeric_limits<double>::min();
    double minValue = std::numeric_limits<double>::max();
    for (size_t i = 0; i < thresholdVector.size(); i++) {
        maxValue = std::max(thresholdVector[i], maxValue);
        minValue = std::min(thresholdVector[i], minValue);
    }

    size_t height = log(indices.size()) + 1;
    tree.nodes_.reserve(pow(2, height) - 1);
    for (size_t i = 0; i < indices.size(); i++) {
        size_t featureIndex = indices[i].first % dataset.featuresNumber();
        size_t thresholdIndex = indices[i].first;
        double threshold = (thresholdVector[thresholdIndex] - minValue) / (maxValue - minValue);
        tree.nodes_.push_back(Node(featureIndex, threshold));
    }
    for (size_t i = indices.size(); i < tree.nodes_.size(); i++) {
        tree.nodes_.push_back(Node(NULL_CLASS));
    }
    std::vector<std::vector<DataItem>> items(tree.nodes_.size());
    for (const DataItem& item : dataset) {
        items[0].push_back(item);
    }
    for (size_t i = 0; i < tree.nodes_.size(); i++) {
        if (items[i].empty()) {
            tree.nodes_[i] = Node(); // null
            continue;
        } else if (items[i].size() <= minItemInLeaf || i >= indices.size()) {
            tree.nodes_[i] = Node(getClass(items[i])); // leaf
            continue;
        }
        for (const DataItem& item : items[i]) {
            bool isLeftNode = tree.nodes_[i].predict(item);
            if (isLeftNode) {
                items[2 * i + 1].push_back(item);
            } else {
                items[2 * i + 2].push_back(item);
            }
        }
    }
    return tree;
}

std::vector<double> ClassificationTree::indexVector() const {
    return indexVector_;
}

std::vector<double> ClassificationTree::thresholdVector() const {
    return thresholdVector_;
}

int ClassificationTree::predict(const DataItem& item) const {
    size_t pos = 0;
    while (!nodes_[pos].isLeaf()) {
        bool isLeftNode = nodes_[pos].predict(item);
        if (isLeftNode) {
            pos = 2 * pos + 1;
        } else {
            pos = 2 * pos + 2;
        }
    }
    return nodes_[pos].y();
}

size_t ClassificationTree::featuresNumber() const {
    return featuresNumber_;
}

void ClassificationTree::save(const std::string& path) const {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("failed to create file: " + path);
    }
    ofs << "features " << featuresNumber() << std::endl;
    for (size_t i = 0; i < nodes_.size(); i++) {
        if (nodes_[i].isLeaf()) {
            ofs << "leaf " << nodes_[i].y() << std::endl;
        } else if (nodes_[i].isNull()) {
            ofs << "null" << std::endl;
        } else {
            ofs << "node " << nodes_[i].featureIndex() << " " << nodes_[i].threshold() << std::endl;
        }
    }
    ofs.close();
}

} // namespace evaboost