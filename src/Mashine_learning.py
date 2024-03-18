from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.datasets import load_iris  # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import yaml



# Abstract base class for classifiers
class Classifier(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def grid_search(self, params: dict, cv: int, X: np.ndarray, y: np.ndarray) -> GridSearchCV:
        pass

# Factory for classifiers
class ClassifierFactory:
    @staticmethod
    def create_classifier(classifier_type: str, grid_params: dict = None, cv: int = 5) -> Classifier:
        """
        Creates a classifier instance with GridSearchCV support.
        
        Args:
            classifier_type (str): Type of classifier ('rf', 'gb', 'xgb', 'dt').
            grid_params (dict): Parameters for GridSearchCV.
            cv (int): Number of cross-validation folds.

        Returns:
            Classifier: An instance of the classifier.
        """
        if classifier_type == 'rf':
            return RandomForestClassifierWrapper(grid_params=grid_params, cv=cv)
        elif classifier_type == 'gb':
            return GradientBoostingClassifierWrapper(grid_params=grid_params, cv=cv)
        elif classifier_type == 'xgb':
            return XGBoostClassifierWrapper(grid_params=grid_params, cv=cv)
        elif classifier_type == 'dt':
            return DecisionTreeClassifierWrapper(grid_params=grid_params, cv=cv)
        else:
            raise ValueError("Unsupported classifier type")



# Wrapper for RandomForest classifier
class RandomForestClassifierWrapper(Classifier):
    def __init__(self, grid_params: dict = None, cv: int = 5):
        self.classifier = RandomForestClassifier(random_state=42)       # Initialization of the RandomForest classifier
        self.grid_params = grid_params                                  # Grid search parameters (candidates)
        self.cv = cv                                                    # Number of cross-validation folds

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifierWrapper':
        if self.grid_params is not None:
            self.grid_search(self.grid_params, self.cv, X, y)           # Perform grid search for the best parameters and train the model
        else:
            self.classifier.fit(X, y)                                   # Train the model with default parameters
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict(X)                               # Make predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(X)                         # Make probability predictions

    def grid_search(self, params: dict, cv: int, X: np.ndarray, y: np.ndarray) -> GridSearchCV:
        grid_search = GridSearchCV(self.classifier, params, cv=cv)      # Grid search for the best parameters
        grid_search.fit(X, y)                                           # Train the model with the best parameters
        self.classifier = grid_search.best_estimator_                   # Set the best parameters
        print("Best RandomForest parameters:", grid_search.best_params_, end='\n')
        return grid_search

# Wrapper for GradientBoosting classifier
class GradientBoostingClassifierWrapper(Classifier):
    def __init__(self, grid_params: dict = None, cv: int = 5):
        self.classifier = GradientBoostingClassifier(random_state=42)   # Initialization of the GradientBoosting classifier
        self.grid_params = grid_params                                  # Grid search parameters (candidates)
        self.cv = cv                                                    # Number of cross-validation folds

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingClassifierWrapper':
        if self.grid_params is not None:
            self.grid_search(self.grid_params, self.cv, X, y)           # Perform grid search for the best parameters and train the model
        else:
            self.classifier.fit(X, y)                                   # Train the model with default parameters
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict(X)                               # Make predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(X)                         # Make probability predictions

    def grid_search(self, params: dict, cv: int, X: np.ndarray, y: np.ndarray) -> GridSearchCV:
        grid_search = GridSearchCV(self.classifier, params, cv=cv)      # Grid search for the best parameters
        grid_search.fit(X, y)                                           # Train the model with the best parameters
        self.classifier = grid_search.best_estimator_                   # Set the best parameters
        print("Best GradientBoosting parameters:", grid_search.best_params_, end='\n')
        return grid_search

# Wrapper for XGBoost classifier
class XGBoostClassifierWrapper(Classifier):
    def __init__(self, grid_params: dict = None, cv: int = 5):
        self.classifier = XGBClassifier()                               # Initialization of the XGBoost classifier
        self.grid_params = grid_params                                  # Grid search parameters (candidates)
        self.cv = cv                                                    # Number of cross-validation folds

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostClassifierWrapper':
        if self.grid_params is not None:
            self.grid_search(self.grid_params, self.cv, X, y)           # Perform grid search for the best parameters and train the model
        else:
            self.classifier.fit(X, y)                                   # Train the model with default parameters
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict(X)                               # Make predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(X)                         # Make probability predictions

    def grid_search(self, params: dict, cv: int, X: np.ndarray, y: np.ndarray) -> GridSearchCV:
        grid_search = GridSearchCV(self.classifier, params, cv=cv)      # Grid search for the best parameters
        grid_search.fit(X, y)                                           # Train the model with the best parameters
        self.classifier = grid_search.best_estimator_                   # Set the best parameters
        print("Best XGBoost parameters:", grid_search.best_params_, end='\n')
        return grid_search
    
# Wrapper for DecisionTree classifier
class DecisionTreeClassifierWrapper(Classifier):
    def __init__(self, grid_params: dict = None, cv: int = 5):
        self.classifier = DecisionTreeClassifier(random_state=42)       # Initialization of the DecisionTree classifier
        self.grid_params = grid_params                                  # Grid search parameters (candidates)
        self.cv = cv                                                    # Number of cross-validation folds

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeClassifierWrapper':
        if self.grid_params is not None:
            self.grid_search(self.grid_params, self.cv, X, y)           # Perform grid search for the best parameters and train the model
        else:
            self.classifier.fit(X, y)                                   # Train the model with default parameters

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict(X)                               # Make predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(X)                         # Make probability predictions

    def grid_search(self, params: dict, cv: int, X: np.ndarray, y: np.ndarray) -> GridSearchCV:
        grid_search = GridSearchCV(self.classifier, params, cv=cv)      # Grid search for the best parameters
        grid_search.fit(X, y)                                           # Train the model with the best parameters
        self.classifier = grid_search.best_estimator_                   # Set the best parameters
        print("Best DecisionTree parameters:", grid_search.best_params_, end='\n')
        return grid_search



# Classifier for ensemble
class EnsembleClassifier(Classifier):
    def __init__(self, classifiers: List[Classifier]):
        self.classifiers = classifiers                                          # List of classifiers

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleClassifier':
        for classifier in self.classifiers:
            classifier.fit(X, y)                                                # Train each classifier
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        probas = np.array([clf.predict_proba(X) for clf in self.classifiers])   # Make probability predictions for each classifier
        avg_probas = np.mean(probas, axis=0)                                    # Average probability predictions
        final_prediction = np.argmax(avg_probas, axis=1)                        # Final prediction
        return final_prediction

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = np.array([clf.predict_proba(X) for clf in self.classifiers])   # Make probability predictions for each classifier
        avg_probas = np.mean(probas, axis=0)                                    # Average probability predictions
        return avg_probas

    def grid_search(self, params: dict, cv: int, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError("Method grid_search is not applicable to ensemble classifiers.")



# Function for loading and processing Iris data
def process_iris_data(test_size: float, random_state: int) -> tuple:
    try:
        # Attempt to load data from a local file
        with open('./data/iris.pkl', 'rb') as f:
            iris = pickle.load(f)
    except FileNotFoundError:
        # Load data if local file is not found
        iris = load_iris()
        with open('./data/iris.pkl', 'wb') as f:
            pickle.dump(iris, f)
    
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test



# Main function to run the program
def main() -> None:
    # Loading configuration from a file
    with open(r'config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    test_size: float    = config['general']['test_size']
    random_state: int   = config['general']['random_state']
    cv: int             = config['general']['cv']
    
    # Processing Iris data
    X_train, X_test, y_train, y_test = process_iris_data(test_size, random_state)
    
    # Creating classifiers
    classifiers: Dict[str, Any] = {
        'Random Forest'     : ClassifierFactory.create_classifier('rf'  , config['classifiers']['RandomForest']['grid_params'], cv),
        'Gradient Boosting' : ClassifierFactory.create_classifier('gb'  , config['classifiers']['GradientBoosting']['grid_params'], cv),
        'XGBoost'           : ClassifierFactory.create_classifier('xgb' , config['classifiers']['XGBoost']['grid_params'], cv),
        'Decision Tree'     : ClassifierFactory.create_classifier('dt'  , config['classifiers']['DecisionTree']['grid_params'], cv)
    }
    
    # Training and evaluating each classifier
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"{name} accuracy: {accuracy:.2f}", end='\n\n')
    
    # Displaying classification report and confusion matrix for the last classifier
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

if __name__ == "__main__":
    main()
