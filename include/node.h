// Copyright 2019 Dolotov Evgeniy

#ifndef EVABOOST_NODE_H
#define EVABOOST_NODE_H

#include <data_item.h>

#include <memory>

namespace evaboost {

class Node {
public:
    Node();

    Node(int y);

    Node(size_t featureIndex, double threshold);

    bool predict(const DataItem& item) const;

    bool isLeaf() const;

    bool isNull() const;

    size_t featureIndex() const;

    double threshold() const;

    int y() const;

private:
    int y_;
    size_t featureIndex_;
    double threshold_;
    bool isLeaf_;
    bool isNull_;
};

} // namespace evaboost

#endif // EVABOOST_NODE_H