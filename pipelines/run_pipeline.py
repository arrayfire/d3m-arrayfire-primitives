from d3m.metadata import base as metadata_base, hyperparams as hyperparams_module, pipeline as pipeline_module, problem
from d3m.container.dataset import Dataset
from d3m.runtime import Runtime

import os
import time

problem_path = '/home/mark/Documents/d3m/d3m_data/datasets/seed_datasets_current/185_baseball/185_baseball_problem/'
dataset_path = '/home/mark/Documents/d3m/d3m_data/datasets/seed_datasets_current/185_baseball/185_baseball_dataset/'
pipeline_dir_path = '/home/mark/Documents/d3m/d3m-arrayfire-primitives/pipelines/'
pipeline_file = 'd1523250-1597-4f71-bebf-738cb6e58217.json'

# Loading problem description.
problem_description = problem.parse_problem_description(problem_path + 'problemDoc.json')

# Loading dataset.
path = 'file://{uri}'.format(uri=os.path.abspath(dataset_path + 'datasetDoc.json'))
dataset = Dataset.load(dataset_uri=path)

# Loading pipeline description file.
with open(pipeline_dir_path + pipeline_file, 'r') as file:
    pipeline_description = pipeline_module.Pipeline.from_json(string_or_file=file)
    # pipeline_description = pipeline_module.Pipeline.from_yaml(string_or_file=file)

# Creating an instance on runtime with pipeline description and problem description.
runtime = Runtime(pipeline=pipeline_description, problem_description=problem_description, context=metadata_base.Context.TESTING)

# Fitting pipeline on input dataset.
t0 = time.time()
fit_results = runtime.fit(inputs=[dataset])
t1 = time.time()
dt_train = t1 - t0
fit_results.check_success()

# Producing results using the fitted pipeline.
t0 = time.time()
produce_results = runtime.produce(inputs=[dataset])
t1 = time.time()
dt_predict = t1 - t0
produce_results.check_success()

print(produce_results.values)
print('Training time: {0:4.4f} s'.format(dt_train))
print('Prediction time: {0:4.4f} s'.format(dt_predict))
