# BANK CREDIT RISK PREDICTION

### Problem Statement
Bank Credit Risk Prediction using South Gemran Credit Dataset.

Build a model to predict whether the person, described by the attributes of the dataset, is a good (1) or a bad (0) credit risk for the bank.

Link to Dataset : https://archive.ics.uci.edu/ml/datasets/South+German+Credit
### Solution Proposed 
The solution model trains a Gradient Boosting Classifier model to classify if a person is a credit risk or not.
## Tools Stack Used
1. Mage AI
2. AWS


## How to run?
Before we run the project, make sure that you are having Python installed and conda configured. You also need AWS account to access S3, EC2 Services.


## Project Architecture
![image](https://github.com/bsb4018/bank_credit_risk_mage/blob/main/assets/file_structure-pipeline.png)


### Step 1: Clone the repository
```bash
git clone https://github.com/bsb4018/activity_pred_main_proj.git
```

### Step 2- Create a conda environment after opening the repository

```bash
conda create -p venv python=3.8 -y
```

### Step 3- Activate the conda environment
```bash
conda activate venv/
```

### Step 4 - Install mage ai
```bash
pip install mage-ai
```

### Step 5 - Create new project and launch tool
```bash
mage start bank_credit_riskp
```

### Step 6 -  
Open http://localhost:6789 in your browser to see the built pipeline.

### Step 7 - Create AWS Account and do the following get the following ids
```bash
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION_NAME
```

### Step 8 - Export the environment variable(LINUX) or Put in System Environments(WINDOWS)
```bash
export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>

export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>

export AWS_REGION_NAME=<AWS_REGION_NAME>

```

### Step 9 - Create S3 buckets to store data and logs
Take the data from data/ and store it in the bucket.
Put the bucket name in bank_credit_riskp/data_loaders/load_data_s3.py under "bucket_name"

Create a bucket to store logs and inside the bucket create a folder named logs
Put the bucket name in bank_credit_riskp/metadata.yaml  logging_config: -> destination_config: -> bucket:

### Step 10 - Create artifacts folder to store model artifacts locally
Create a folder named "artifacts" in the same lcoation as the repository and two sub folders named "model" and "data-split" 


### Step 11 - Trigger Pipeline in Mage UI to execute or run the following
```bash
mage run bank_credit_riskp bank_criskp_aws
```
