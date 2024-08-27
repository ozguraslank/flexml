![Python versions](https://img.shields.io/badge/python_3.9+-blue)
# FlexML

<div align="center">
<img src="img/flexml_banner.jpeg" alt="drawing" width="400"/>
</div>

## Introduction
FlexML is an easy-to-use and flexible AutoML library for Python that simplifies the process of building machine learning models. It automates model selection and hyperparameter tuning, offering users the flexibility to customize the size of their experiments by allowing to train all available models in the library or only a subset of them for faster results, FlexML adapts to your needs! <br> <br>

At the moment, FlexML supports only regression and classification tasks and offers two experiment modes; 'quick' and 'wide' allowing users to choose between fitting a few of machine learning models or the full range of models available in the library. This flexibility extends to hyperparameter tuning as well, enabling a balance between speed and thoroughness.

## How to Install
To install FlexML, you can use pip:

```bash
pip install flexml
```

## Quick Start

```python
# Experiment for a Regression problem for diabates dataset
from flexml import Regression
from sklearn.datasets import load_diabetes

# Load the diabetes dataset
df = load_diabetes(as_frame=True)['frame']

# Initialize the regression experiment
reg_exp = Regression(df, target_col="target")

# Start the experiment with 'quick' experiment size and r2 evaluation metric
reg_exp.start_experiment(exp_size="quick", eval_metric="r2")

# Get the best model
best_model = reg_exp.get_best_models(top_n_models=1)
print(best_model)
>>> LinearRegression()

# Tune the best model with Optuna
reg_exp.tune_model(tuning_method="optuna", tuning_size="wide", eval_metric="r2")
tuned_model = reg_exp.tuned_model
print(tuned_model)
>>> LinearRegression()
```
You can also take a look to jupyter notebook files in the 'notebooks' folder in the repository for more detailed explanations of the usage

## How to Contribute:

1. **Fork the repository:** Click on the 'Fork' button at the top right corner of the GitHub repository page
2. **Create a new branch:** Name your branch descriptively based on the feature or fix you're working on
3. **Make your changes:** Write code and tests to add your feature or fix the issue.
   - You can take a look to **tests** folder in the repository to reach the current unittests
4. **Run tests:** Ensure all existing and new tests pass.
5. **Submit a pull request:** Open a pull request with a clear description of your changes.

## Roadmap
- [x] Regression
- [x] Classification
- [x] Model Tuning
- [ ] Automatic Data Labelling & Feature Engineering
- [ ] Time Series  
