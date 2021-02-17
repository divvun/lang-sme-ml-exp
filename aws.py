import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

sagemaker.Session(boto3.session.Session())


pytorch_estimator = PyTorch('train.py',
                            instance_type='ml.g4dn.xlarge',
                            instance_count=1,
                            framework_version='1.6',
                            py_version='py3',
                            # use_spot_instances=True,
                            role='arn:aws:iam::125752969235:role/service-role/AmazonSageMaker-ExecutionRole-20210215T171358',
                            hyperparameters = {'epochs': 1, 'batch-size': 64})
pytorch_estimator.fit({'train': 's3://thetc-ml-1/data/'})