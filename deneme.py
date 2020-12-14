# -*- coding: utf-8 -*-
"""
Created on Sat May 18 00:34:59 2019

@author: furkan
"""

import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.grid.grid_search import H2OGridSearch
from xlrd import open_workbook
from openpyxl.reader.excel import load_workbook
import csv
h2o.init()
h2o.remove_all() 

train = h2o.import_file("./higgs_train_10k.csv")
test = h2o.import_file("./higgs_test_5k.csv")

x = train.columns
y = "response"
x.remove(y)

train[y] = train[y].asfactor()
test[y] = test[y].asfactor()
nfolds = 5



my_gbm = H2OGradientBoostingEstimator(distribution="bernoulli",
                                      ntrees=10,max_depth=3,
                                      min_rows=2,learn_rate=0.2,nfolds=nfolds,
                                      fold_assignment="Modulo",keep_cross_validation_predictions=True,seed=1)
my_gbm.train(x=x, y=y, training_frame=train)


my_rf = H2ORandomForestEstimator(ntrees=50,
                                 nfolds=nfolds,
                                 fold_assignment="Modulo",
                                 keep_cross_validation_predictions=True,
                                 seed=1)
my_rf.train(x=x, y=y, training_frame=train)


ensemble = H2OStackedEnsembleEstimator(model_id="my_ensemble_binomial",
                                       base_models=[my_gbm.model_id, my_rf.model_id])
ensemble.train(x=x, y=y, training_frame=train)



perf_stack_test = ensemble.model_performance(test)

perf_gbm_test = my_gbm.model_performance(test)


perf_rf_test = my_rf.model_performance(test)


baselearner_best_auc_test = max(perf_gbm_test.auc(), perf_rf_test.auc())
stack_auc_test = perf_stack_test.auc()
print("Best Base-learner Test AUC:  {0}".format(baselearner_best_auc_test))
print("Ensemble Test AUC:  {0}".format(stack_auc_test))



pred = ensemble.predict(test)


hyper_params = {"learn_rate": [0.01, 0.03,0.1,0.3,0.2],
                "max_depth": [3, 4, 5, 6, 9],
                "sample_rate": [0.7, 0.8, 0.9, 1.0],
                "col_sample_rate": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9]}
search_criteria = {"strategy": "RandomDiscrete", "max_models": 3, "seed": 2}

grid = H2OGridSearch(model=H2OGradientBoostingEstimator(ntrees=10,seed=1,nfolds=nfolds,fold_assignment="Modulo"
                                                      ,keep_cross_validation_predictions=True),
hyper_params=hyper_params,search_criteria=search_criteria,grid_id="gbm_grid_binomial")
grid.train(x=x, y=y, training_frame=train)



# Train a stacked ensemble using the GBM grid
ensemble = H2OStackedEnsembleEstimator(model_id="my_ensemble_gbm_grid_binomial",
                                       base_models=grid.model_ids)
ensemble.train(x=x, y=y, training_frame=train)

perf_stack_test = ensemble.model_performance(test)

baselearner_best_auc_test = max([h2o.get_model(model).model_performance(test_data=test).auc() for model in grid.model_ids])
baselearner_best_auc_test2 = [h2o.get_model(model).model_performance(test_data=test).auc() for model in grid.model_ids]

stack_auc_test = perf_stack_test.auc()



print("Best Base-learner Test AUC:  {0}".format(baselearner_best_auc_test))
print("Best Base-learner Test AUC:  {0}".format(baselearner_best_auc_test2))
print("Ensemble Test AUC:  {0}".format(stack_auc_test))



pred = ensemble.predict(test)
pred2 = ensemble.fit(train)
print(pred2)


