// Copyright 2019 Dolotov Evgeniy

#ifndef EVABOOST_DATASET_H
#define EVABOOST_DATASET_H

#include <data_item.h>

#include <set>
#include <string>
#include <vector>
#include <unordered_map>

namespace evaboost {

class Dataset {
public:
    using iterator = std::vector<DataItem>::iterator;
    using const_iterator = std::vector<DataItem>::const_iterator;

public:
    // empty dataset
    Dataset() = default;

    // CSV without header
    static Dataset loadFromCSV(const std::string& path, size_t yPos);

    // CSV with header
    static Dataset loadFromCSV(const std::string& path, const std::string& yName);

    iterator begin();

    const_iterator begin() const;

    const_iterator cbegin() const;

    iterator end();

    const_iterator end() const;

    const_iterator cend() const;

    size_t itemsNumber() const;

    DataItem itemAt(size_t i) const;

    size_t featuresNumber() const;

    size_t classesNumber() const;

    void addItem(const DataItem& item);

    Dataset sampleDataset(size_t datasetSize) const;

private:
    std::vector<DataItem> items_;
    std::set<int> classes_;
};

} // namespace evaboost

#endif // EVABOOST_DATASET_H