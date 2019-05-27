#include <data_item.h>

#include <string>
#include <exception>
#include <stdexcept>

namespace evaboost {

DataItem::DataItem(const std::vector<double>& features, int y)
    : features_(features),
      y_(y)
{}

const std::vector<double>& DataItem::features() const {
    return features_;
}

double DataItem::featureAt(size_t i) const {
    if (i < features_.size()) {
        return features_[i];
    } else {
        throw std::runtime_error("Invalid feature index:" + std::to_string(i));
    }
}

size_t DataItem::featuresNumber() const {
    return features_.size();
}

int DataItem::y() const {
    return y_;
}

} // namespace evaboost