#include <node.h>

#include <exception>
#include <stdexcept>

namespace evaboost {

Node::Node()
    : isLeaf_(false),
      isNull_(true)
{}

Node::Node(int y)
    : y_(y),
      isLeaf_(true),
      isNull_(false)
{}

Node::Node(size_t featureIndex, double threshold)
    : y_(NULL_CLASS),
      featureIndex_(featureIndex),
      threshold_(threshold),
      isLeaf_(false),
      isNull_(false)
{}

bool Node::predict(const DataItem& item) const {
    return item.featureAt(featureIndex_) <= threshold_;
}

bool Node::isLeaf() const {
    return isLeaf_;
}

bool Node::isNull() const {
    return isNull_;
}

int Node::y() const {
    if (isLeaf()) {
        return y_;
    } else {
        throw std::runtime_error("Node is not leaf");
    }
}

size_t Node::featureIndex() const {
    return featureIndex_;
}

double Node::threshold() const {
    return threshold_;
}

} // namespace evaboost