# A study of RayS Attack on Vision Transformer Robustness using CIFAR10 Dataset


 🧑‍💻 Venkata Abhiram Chitty

🏣 **Affiliation:** Kennesaw State Univeristy, Kennesaw, GA

This repository is the reproduction of ICCV 2021 paper ["On the Robustness of Vision Transformers to Adversarial Examples"](https://openaccess.thecvf.com/content/ICCV2021/papers/Mahmood_On_the_Robustness_of_Vision_Transformers_to_Adversarial_Examples_ICCV_2021_paper.pdf) paper based on the author's offical GitHub [repo](https://github.com/MetaMain/ViTRobust/tree/main). 




## Getting Started

Create a Virtual Environment
```bash
conda create -n ViT_Robustness python=3.10
conda activate ViT_Robustness
```

Install Requirements
```bash
pip install -r requirements.txt 
```

The models can be downloaded [here](https://uconn-my.sharepoint.com/:f:/g/personal/kaleel_mahmood_uconn_edu/EgkVWxG7wb5FlUkzDRhFMC4B_LawJR0cBsn92GZeyZ3lDg?e=pSS3QK).

Example to run ***ViT-L_16*** Model on ***100*** CIFAR10 Dataset Images with epsMax value of ***0.031*** on queryLimit of ***3000*** 
```bash
python main.py \
       --model_name "ViT-L_16" \
       --epsMax 0.031 \
       --num_clean_images 100 \
       --numClasses 10 \
       --queryLimit 3000 \
```


## Results

| Model | Eps Max | Queries Used | Robust Accuracy | Clean Accuracy |
|-------|---------|--------------|-----------------|----------------|
| ViT-L_16  | 0.031 | 3000 | 0.47 | 0.991 |
| ViT-L_16  | 0.062 | 3000 | 0.26 | 0.991 |


