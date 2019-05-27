// Copyright 2019 Dolotov Evgeniy

#include <dataset.h>
#include <data_item.h>
#include <evaluation.h>
#include <classification_tree.h>

#include <cmath>
#include <exception>
#include <stdexcept>

namespace evaboost {

namespace {

const int NEGATIVE_CLASS = 0;
const int POSITIVE_CLASS = 1;

    void evaluate(
        const Dataset& dataset, const ClassificationTree& tree,
        size_t& truePositive, size_t& trueNegative,
        size_t& falsePositive, size_t& falseNegative)
{
    if (dataset.classesNumber() != 2) {
        throw std::runtime_error("Classes number should be equal to 2");
    }
    truePositive = 0;
    trueNegative = 0;
    falsePositive = 0;
    falseNegative = 0;
    for (const DataItem& item : dataset) {
        int yPredict = tree.predict(item);
        if (item.y() == NEGATIVE_CLASS) {
            if (NEGATIVE_CLASS == yPredict) {
                trueNegative++;
            } else if (POSITIVE_CLASS == yPredict) {
                falsePositive++;
            } else {
                throw std::runtime_error("Invalid predicted class");
            }
        } else if (item.y() == POSITIVE_CLASS) {
            if (POSITIVE_CLASS == yPredict) {
                truePositive++;
            } else if (NEGATIVE_CLASS == yPredict) {
                falseNegative++;
            } else {
                throw std::runtime_error("Invalid predicted class");
            }
        } else {
            throw std::runtime_error("Invalid class in dataset");
        }
    }
}

double recall(size_t truePositive, size_t falseNegative) {
    return truePositive / double(truePositive + falseNegative);
}

double precision(size_t truePositive, size_t falsePositive) {
    return truePositive / double(truePositive + falsePositive);
}

} // namespace


double accuracy(const Dataset& dataset, const ClassificationTree& tree) {
    size_t correctCount = 0;
    for (const DataItem& item : dataset) {
        int yPredict = tree.predict(item);
        if (item.y() == yPredict) {
            correctCount++;
        }
    }
    return correctCount / (double)dataset.itemsNumber();
}

double recall(const Dataset& dataset, const ClassificationTree& tree) {
    size_t tp, tn, fp, fn;
    evaluate(dataset, tree, tp, tn, fp, fn);
    return recall(tp, fn);
}

double precision(const Dataset& dataset, const ClassificationTree& tree) {
    size_t tp, tn, fp, fn;
    evaluate(dataset, tree, tp, tn, fp, fn);
    return precision(tp, fp);
}

double f1(const Dataset& dataset, const ClassificationTree& tree) {
    size_t tp, tn, fp, fn;
    evaluate(dataset, tree, tp, tn, fp, fn);
    return 2 * recall(tp, fn) * precision(tp, fp) / (recall(tp, fn) + precision(tp, fp));
}

double mcc(const Dataset& dataset, const ClassificationTree& tree) {
    size_t tp, tn, fp, fn;
    evaluate(dataset, tree, tp, tn, fp, fn);
    return (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
}

} // namespace evaboost