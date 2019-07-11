from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

# Creating pipeline
pipeline_description = Pipeline()
pipeline_description.add_input(name='inputs')

# Try loading the logit primitive without doing anything with it yet
logit_step = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.regression.logistic.AFPrimitives'))
print('Logit primitive discovered!')
