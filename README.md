# MLops-Project

## What is done?
1. In this project i've trained and wrapped fast.ai classifier(cat/dog) with simplest RESTful API(FastAPI) into docker container.
2. Added an version control for model and data with DVC.
3. Added additional class label Rabbit, scrapped from internet and retrained model on it.
4. Created couple of pipelines using Google Cloud Platform + Airflow (Stored in DAGs folder)
   - Sort input files by folders
   - Inference using that images as models input
   - Finetune model using data selected by user(moved to specific folder).
