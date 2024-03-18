# Iris Plant Classification Module

## Overview
This project is a comprehensive solution for classifying Iris plants from the dataset using various machine learning algorithms. It includes support for several classification models: Random Forest, Gradient Boosting, XGBoost, and Decision Tree. Developed using OOP principles and the Factory pattern for ease of extension and integration.

## Requirements
- Python 3.8+ (compatibility with other Python versions is not guaranteed).
- Required libraries:
  ```
  scikit-learn
  numpy
  xgboost
  pickle
  yaml
  ```
  To install all libraries at once, run the command: `pip install -r requirements.txt`.

## Installation
1. Ensure that Python 3.8+ is installed on your system.
2. Clone this repository or download the project files to your computer:
   ```
   git clone https://github.com/Ceslavas/mashine_learning.git "D:\your_mashine_learning_folder"
   ```

## Configuration
Before using the classification module, set up the `config.yaml` file:
1. Specify the parameters for training and testing the models, including the test size and random state.
2. Example `config.yaml` file:
```yaml
general:
  test_size: 0.2
  random_state: 42
  cv: 5
classifiers:
  RandomForest:
    grid_params: {...}
  GradientBoosting:
    grid_params: {...}
  XGBoost:
    grid_params: {...}
  DecisionTree:
    grid_params: {...}
```

## Running the Project
To use the classification module, follow these steps:
1. Open a command line or terminal.
2. Navigate to the directory where the `src\Mashine_learning.py` script is located.
3. Enter the command `python Mashine_learning.py`.

## Results
The scripts will process the input data according to the specified configuration, performing model training, evaluation, and outputting the results to the console.

## FAQ
**Q:** Can the module be used to process other datasets?
**A:** Yes, the module can be adapted to work with other datasets, only the data preparation needs to be in accordance with the expected format.

## Contributing
Contributions to the project are welcome! If you have suggestions for improvements or new features, please submit a pull request or create an issue.

## License
This project is distributed under the MIT license. See the LICENSE.txt file for details.
