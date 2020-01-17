# ArrayFire Accelerated Primitives

DARPA D3M TA1 primitives written by ArrayFire. These primitives are built such
that it utilizes the GPU to accelerate computation by using the
[arrayfire-python](https://github.com/arrayfire/arrayfire-python) package.

## Quick Start

The following command assumes that you have the following:
- [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) installed
- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) installed
- A location in your file system that a D3M docker container can bind-mount and
  access
- This repository accessible from the docker bind-mount location
- The D3M datasets repository accessible from the docker bind-mount location
  (this can be cloned from the [public D3M datasets](
  https://datasets.datadrivendiscovery.org/d3m/datasets) or the internal version
  of it)
- Inside the D3M datasets repository, the `185_baseball` dataset's
  `learningData.csv` downloaded through git LFS (see the `README` of the
  datasets repo)
  
Change the following in the command according to your environment:
- `-v` parameter of `docker run`
- `-r, -i, -t, -a, -p` parameters of `python3 -m d3m runtime fit-score`

```shell
docker run \
    --gpus all \
    --rm \
    -v /home/mark/Documents/d3m:/mnt/d3m \
    registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9 \
    /bin/bash -c \
        "pip3 install -e /mnt/d3m/d3m-arrayfire-primitives && \
        LD_PRELOAD=/opt/arrayfire/lib64/libafcuda.so \
        LD_LIBRARY_PATH=/opt/arrayfire/lib64 \
        python3 -m d3m \
            runtime \
            fit-score \
                -r /mnt/d3m/datasets/seed_datasets_current/185_baseball/185_baseball_problem/problemDoc.json \
                -i /mnt/d3m/datasets/seed_datasets_current/185_baseball/TRAIN/dataset_TRAIN/datasetDoc.json \
                -t /mnt/d3m/datasets/seed_datasets_current/185_baseball/TEST/dataset_TEST/datasetDoc.json \
                -a /mnt/d3m/datasets/seed_datasets_current/185_baseball/SCORE/dataset_TEST/datasetDoc.json \
                -p /mnt/d3m/d3m-arrayfire-primitives/pipelines/classification.logistic_regression.ArrayFire/d1523250-1597-4f71-bebf-738cb6e58217.json"
```

Notice that for now, `LD_PRELOAD` and `LD_LIBRARY_PATH` has to be defined for
the d3m runtime to correctly load the ArrayFire libraries.

## Using these primitives on image datasets

These primitives can be used on image datasets (such as `124_120_mnist`), but
the pipeline for accomplishing this must use the
[`data_preprocessing.ndarray_flatten.Common`
primitive](https://gitlab.com/mark-poscablo/common-primitives/blob/ndarray-flatten/common_primitives/ndarray_flatten.py)
as a step after `data_preprocessing.image_reader.Common`. This primitive is
currently in a [merge
request](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/109)
to the d3m `common-primitives` repo. Note that this is necessary as well for
other primitives (such as `sklearn-wrap`'s version of the logit classifier
primitive) when dealing with image datasets. See the
[`90630b51-52b1-4439-b1bf-eb470d6a88b4.yml`
pipeline](https://github.com/arrayfire/d3m-arrayfire-primitives/blob/master/pipelines/classification.logistic_regression.ArrayFire/90630b51-52b1-4439-b1bf-eb470d6a88b4.yml)
for an example.
