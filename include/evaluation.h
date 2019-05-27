// Copyright 2019 Dolotov Evgeniy

#ifndef EVABOOST_EVALUATION_H
#define EVABOOST_EVALUATION_H

#include <dataset.h>
#include <classification_tree.h>

namespace evaboost {

double accuracy(const Dataset& dataset, const ClassificationTree& tree);

double recall(const Dataset& dataset, const ClassificationTree& tree);

double precision(const Dataset& dataset, const ClassificationTree& tree);

double f1(const Dataset& dataset, const ClassificationTree& tree);

double mcc(const Dataset& dataset, const ClassificationTree& tree);

} // namespace evaboost

#endif // EVABOOST_EVALUATION_H