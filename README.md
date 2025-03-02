# ViT-Robustness
A study of RayS Attack on Vision Transformer Robustness using CIFAR10 Dataset
🧑‍💻 Venkata Abhiram Chitty

🏣 Affiliation: Kennesaw State Univeristy, Kennesaw, GA

This repository is the reproduction of ICCV 2021 paper "On the Robustness of Vision Transformers to Adversarial Examples" paper based on the author's offical GitHub repo.

Getting Started
Create a Virtual Environment

conda create -n ViT_Robustness python=3.10
conda activate ViT_Robustness
Install Requirements

pip install -r requirements.txt 
The models can be downloaded here.

Example to run ViT-L_16 Model on 100 CIFAR10 Dataset Images with epsMax value of 0.031 on queryLimit of 3000

python main.py \
       --model_name "ViT-L_16" \
       --epsMax 0.031 \
       --num_clean_images 100 \
       --numClasses 10 \
       --queryLimit 3000 \
Results
Model	Eps Max	Queries Used	Robust Accuracy	Clean Accuracy
ViT-L_16	0.031	3000	0.47	0.991
ViT-L_16	0.062	3000	0.26	0.991


