// Copyright 2019 Dolotov Evgeniy

#ifndef EVABOOST_DATA_ITEM_H
#define EVABOOST_DATA_ITEM_H

#include <vector>

namespace evaboost {

const int NULL_CLASS = -1;

class DataItem {
public:
    DataItem(const std::vector<double>& features, int y = NULL_CLASS);

    const std::vector<double>& features() const;

    double featureAt(size_t i) const;

    size_t featuresNumber() const;

    int y() const;

private:
    std::vector<double> features_;
    int y_;
};

} // namespace evaboost

#endif // EVABOOST_DATA_ITEM_H