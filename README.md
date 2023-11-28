# product-quality-ai

## Application

The Product Quality is currently used in the Pack Plan. It takes in images in the Pack Plan and give a score between 0-10 on the product quality, where 10 is the best quality. It uses computer vision to **detect the defects on the fruit shell**. This is a general model that is used regardless of product category and SKU. 

### Rationale 

A general model is chosen for Product Quality because it is too costly to develop a model for each product category. We previously tried a pineapple shell colour model but that only applied to pineapples (ripeness varies per fruit so it is not applicable to other fruits).

### Future Development

1. Since the current development is a general model, a product-specific model can be developed.
2. Other potential development areas include defect model for other fresh produce categories / varieties. The model was last trained in Dec 2022 and there have been more trades tracked on the platform.
3. Integrate the Product Quality into other features on the DiMuto platform.

## Set up

1. Create Python 3.9 virtual environment
2. Activate virtual environment
3. Install requirements.txt
3. Add files with AWS access keys in `src` folder 

`src/keys.py`
"""
ACCESS_ID = XXX
ACCESS_KEY = XXX
"""

* For access keys, refer to AI Coda 

## Production

-  Deployed: AWS ECS arn:aws:ecs:ap-southeast-1:555887774394:cluster/ai-production-cluster
    * A docker container is deployed (`Dockerfile`)
-  Development: AWS EC2
  * i-0dd666148efa6d451 (product-quality-ai-production) → small instance, used for code change
  * i-046b602ef8dd02133 (product-quality-ai-gpu) → large instance, used for model training
- Pem file: dimuto-ai.pem

## AI Model

### Model Architecture

The computer vision model consists of two components:
(1) Defect detection model to detect the fruits in a pack
(2) Classification model to classify whether there are defects in the fruits

### Defect Calculation

Score of individual fruit from classification model -> Average score of the fruits in the box -> Average score of the boxes in the pack plan

### Model Training

The model uses images from common fruits like apples, pineapples and bananas. These images can be found in the s3 bucket (product-quality-ai).

### Model Performance

With the validation set, the detection model achieved a score of 78.2% accuracy. The classification model achieved a score of 83.4% accuracy.