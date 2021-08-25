# MLops-Project

## What is done?
1. In this project i've trained and wrapped Fast-AI(cat/dog) classifier with simplest RESTful API into docker container with
2. Added an version control for model and data with DVC.
3. Added additional class label Rabbit, scrapped from internet.
4. Created couple of pipelines using Google Cloud Platform + Airflow
   - Sort input files by folders
   - Inference using that images as models input
   - Finetune model using data selected by user(moved to specific folder).
