// Copyright 2019 Dolotov Evgeniy

#ifndef EVABOOST_CLASSIFICATION_TREE_H
#define EVABOOST_CLASSIFICATION_TREE_H

#include <node.h>
#include <dataset.h>
#include <data_item.h>

#include <string>

namespace evaboost {

class ClassificationTree {
public:
    static ClassificationTree fromCART(const std::string& path);

    static ClassificationTree fromRealVectors(
        const std::vector<double>& indexVector,
        const std::vector<double>& thresholdVector,
        const Dataset& dataset,
        size_t minLeafInNode);

    std::vector<double> indexVector() const;

    std::vector<double> thresholdVector() const;

    int predict(const DataItem& item) const;

    size_t featuresNumber() const;

    void save(const std::string& path) const;

private:
    ClassificationTree() = default;

    void toRealVectors(
        std::vector<double>& indexVector,
        std::vector<double>& thresholdVector) const;

    std::vector<Node> nodes_;
    size_t featuresNumber_;

    std::vector<double> indexVector_;
    std::vector<double> thresholdVector_;
};

} // namespace evaboost

#endif // EVABOOST_CLASSIFICATION_TREE_H