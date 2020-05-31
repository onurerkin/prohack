from dataclasses import dataclass
from typing import Optional

from src.contracts.Dataset import Dataset

"""
TPOTParams is the dataclass we will use if we want to train a tpot automl pipeline.
You can read the documentation on tpot for specific argument on this web page :
https://epistasislab.github.io/tpot/

Here is the meaning of the arguments you will be able to play with in bdcML. We kept the same name as the documentation.

dataset : is the Dataset dataclass that must contains training and validation dataset.

path_to_export: Tpot provide the python code that generate the model pipeline. This python code can be saved. If
you want to save it, you need to provide a path to store the python file. If you fill the parameter with an empty
string, we will not save the python file. (Note: you need to provide the complete path plus the file name ex:
'path/name.py')

scoring: Function used to evaluate the quality of a given pipeline. If you set this variable to an empty string "", we
will set the following default value for them:
- regression : neg_mean_absolute_error
- classification : f1_macro

generations: int, it's the number of iteration you want to run the pipeline optimization process.

population_size: Number of individuals to retain in the genetic programming population every generation. The higher
this value is the better tpot will perform

config_dict: A configuration dictionary for customizing the operators and parameters that TPOT searches in the
optimization process. The possible values are:
-'TPOT light' : will use a built-in configuration with only fast models and preprocessors
-'TPOT MDR', TPOT will use a built-in configuration specialized for genomic studies
-'TPOT sparse':TPOT will use a configuration dictionary with a one-hot encoder and the operators normally included in
TPOT that also support sparse matrices
-None, TPOT will use the default TPOTRegressor configuration. default value

custom_validation: this determine if you want to use the validation dataset instantiated in the dataset argument to
validate the model. If you set False, it will use cross validation using the training dataset.

Notes:
- If you need more control, we can still add more hyper-parameters. But default ones should give good results
- random state is set to 100
- make sure to use a good scoring parameter based on the problem
"""

@dataclass
class TPOTParams:
    dataset: Dataset
    path_to_export: str
    scoring: str = ""
    generations: int = 100
    population_size: int = 20
    config_dict: Optional[str] = None
    custom_validation: bool = True