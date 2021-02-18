import boto3
import sagemaker
import datetime
import logging

from sagemaker.pytorch import PyTorch

boto3.set_stream_logger('sagemaker', logging.DEBUG)
sagemaker.Session(boto3.session.Session())

IMAGE_URI = "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training:1.7.1-gpu-py36-cu110-ubuntu18.04"
ROLE = 'arn:aws:iam::125752969235:role/service-role/AmazonSageMaker-ExecutionRole-20210215T171358'
S3_BUCKET = "s3://thetc-ml-1"
PROJ_NAME = "lang-sme-ml-exp"

date_now = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S-%f")
job_name = f"{PROJ_NAME}-{date_now}"
output_path = f"{S3_BUCKET}/{PROJ_NAME}/"
checkpoint_path = f"{S3_BUCKET}/{PROJ_NAME}/{job_name}/checkpoints/"
code_path = f"{S3_BUCKET}/{PROJ_NAME}/{job_name}/source" # No trailing slash here because AWS is dumb

print("[-] DATA IS BEING GENERATED AT:")
print(f"  {output_path}{job_name}/")

HYPERPARAMS = {
    'epochs': 3,
    'batch-size': 64
}

pytorch_estimator = PyTorch('train.py',
                            instance_type='ml.g4dn.xlarge', #'ml.p3.2xlarge',
                            instance_count=1,
                            image_uri=IMAGE_URI,
                            role=ROLE,
                            disable_profiler=True,
                            output_path=output_path,
                            checkpoint_s3_uri=checkpoint_path,
                            code_location=code_path,
                            hyperparameters = HYPERPARAMS,
                            use_spot_instances = True,
                            max_wait = 86400)

pytorch_estimator.fit(
    {'train': f'{S3_BUCKET}/{PROJ_NAME}/data/'},
    job_name=job_name
)

final_path = f"{output_path}{job_name}/output/model.tar.gz"
print("[-] FINAL MODEL UPLOADED TO:")
print(f"  {final_path}")
print()
print(f"Run `aws s3 cp {final_path} .` to download to current directory.")
