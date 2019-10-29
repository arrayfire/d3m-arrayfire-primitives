from d3m.metadata import base as metadata_base, hyperparams as hyperparams_module, pipeline as pipeline_module, problem
from d3m.container.dataset import Dataset
from d3m.runtime import Runtime

import os
import time
import logging

logging.basicConfig()

# dataset_name = '185_baseball'
# dataset_name = '32_wikiqa'
# dataset_name = '124_174_cifar10'
# dataset_name = '124_214_coil20'
# problem_path = '/home/mark/Documents/d3m/d3m_data/datasets/seed_datasets_current/' + dataset_name + '/' + dataset_name + '_problem/'
# dataset_path = '/home/mark/Documents/d3m/d3m_data/datasets/seed_datasets_current/' + dataset_name + '/' + dataset_name + '_dataset/'

dataset_name = '124_120_mnist'
# dataset_name = '124_120_mnist_small'
problem_path = '/home/mark/Documents/d3m/d3m_data/datasets/training_datasets/' + dataset_name + '/' + dataset_name + '_problem/'
dataset_path = '/home/mark/Documents/d3m/d3m_data/datasets/training_datasets/' + dataset_name + '/' + dataset_name + '_dataset/'

# # sklearn logit classifier
# pipeline_dir_path = '/home/mark/Documents/d3m/primitives/v2019.6.7/JPL/d3m.primitives.classification.logistic_regression.SKlearn/2019.6.7/pipelines/'
# pipeline_file = '6a68e117-5e2c-4959-90ee-55ad86354fad.json'

# sklearn keraswrap
# pipeline_dir_path = '/home/mark/Documents/d3m/primitives/v2019.6.7/JPL-manual/d3m.primitives.learner.model.KerasWrap/0.2.0/pipelines/'
# pipeline_file = '32418dc0-a64d-4cc6-af3f-671a828a22e3.yml' # sk logit

pipeline_dir_path = '/home/mark/Documents/d3m/d3m-arrayfire-primitives/pipelines/'
# pipeline_file = 'd1523250-1597-4f71-bebf-738cb6e58217.json' # af logit
# pipeline_file = '4917e710-8361-43ab-9dc7-6870b351d5b6.json' # af_knn
pipeline_file = '90630b51-52b1-4439-b1bf-eb470d6a88b4.yml'

# Loading problem description.
problem_description = problem.parse_problem_description(problem_path + 'problemDoc.json')

# Loading dataset.
path = 'file://{uri}'.format(uri=os.path.abspath(dataset_path + 'datasetDoc.json'))
dataset = Dataset.load(dataset_uri=path)

# Loading pipeline description file.
with open(pipeline_dir_path + pipeline_file, 'r') as file:
    if pipeline_file.endswith('json'):
        pipeline_description = pipeline_module.Pipeline.from_json(string_or_file=file)
    else:
        pipeline_description = pipeline_module.Pipeline.from_yaml(string_or_file=file)

# Creating an instance on runtime with pipeline description and problem description.
runtime = Runtime(pipeline=pipeline_description,
                  problem_description=problem_description,
                  context=metadata_base.Context.TESTING,
                  is_standard_pipeline=True)

print('Starting fit...')
import pdb; pdb.set_trace()
fit_results = runtime.fit(inputs=[dataset], return_values=['outputs.0'])

# # Fitting pipeline on input dataset.
# t0 = time.time()
# fit_results = runtime.fit(inputs=[dataset], return_values=['outputs.0'])
# t1 = time.time()
# dt_train = t1 - t0
# fit_results.check_success()
# print('Training time: {0:4.4f} s'.format(dt_train))

# # Producing results using the fitted pipeline.
# t0 = time.time()
# produce_results = runtime.produce(inputs=[dataset])
# t1 = time.time()
# dt_predict = t1 - t0
# produce_results.check_success()
# # print(produce_results.values)
# print('Prediction time: {0:4.4f} s'.format(dt_predict))
