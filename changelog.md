# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1] - 2023-05-03

### Removed

- The requirement.txt file from the `MSC APP` folder
- Unused normalize.css file
- Identical links assigned in each translation file
- Duplicate index file for the english version

### Added

- `import nltk` in `helper.py`
- `nltk.download('stopwords')` in `helper.py`
- `nltk.download('punkt')` in `helper.py`
- directory `pickles` under `MSC APP`
- The requirement.txt file to the root directory of the project

### Changed

- Upgrade dependency: numpy to 1.24.3

## [0.2] - 2023-05-06

### Added

- BNB classifier

## [1.0] - 2023-05-12

### Added

- MNB classifier
- UI for customizing Naive Bayes parameters

## [1.1] - 2023-05-16

### Added

- Functionality to save the vectorization into pickle file to ensure feature consistency

## [1.2] - 2023-05-19

### Added

- UI for Transfer / Incremental Learning

## [2.0] - 2023-05-24

### Added

- One new dataset - Kaggle 2 dataset
- Python script which takes the first 3000 rows of Kaggle 2
- The resulting new dataset