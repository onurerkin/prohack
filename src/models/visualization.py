# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style
from pandas_profiling import ProfileReport
import pandas_profiling




X_train_visual=X_train.iloc[:, : 15]

profile = ProfileReport(X_train_visual)

profile.to_file("your_report.html")
