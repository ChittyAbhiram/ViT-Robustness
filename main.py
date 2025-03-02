

import torch
import numpy
from ModelPlus import ModelPlus
import DataManagerPytorch as DMP
import AttackWrappersRayS
from TransformerModels import VisionTransformer, CONFIGS
from collections import OrderedDict
import argparse

def LoadViTLAndCIFAR10(model_name, numClasses, batchSize, imgSize):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valLoader = DMP.GetCIFAR10Validation(imgSize, batchSize)
    config = CONFIGS[model_name]
    model = VisionTransformer(config, imgSize, zero_head=True, num_classes=numClasses)
    dir = "ViT-L_16,cifar10,run0_15K_checkpoint.bin"
    dict = torch.load(dir)
    model.load_state_dict(dict)
    model.eval()
    modelPlus = ModelPlus(model_name, model, device, imgSizeH=imgSize, imgSizeW=imgSize, batchSize=batchSize)
    return valLoader, modelPlus

def RaySAttackVisionTransformer(model_name, numClasses, batchSize, imgSize, attackSampleNum, epsMax, queryLimit, num_clean_images):

    valLoader, defense = LoadViTLAndCIFAR10(model_name, numClasses, batchSize, imgSize)
    cleanLoader = DMP.GetCorrectlyIdentifiedSamplesBalancedDefense(defense, attackSampleNum, valLoader, numClasses)
    advLoader = AttackWrappersRayS.RaySAttack(defense, epsMax, queryLimit, cleanLoader, num_clean_images)
    robustAcc = defense.validateD(advLoader)
    cleanAcc = defense.validateD(valLoader)

    print("Queries used:", queryLimit)
    print("Robust acc:", robustAcc)
    print("Clean acc:", cleanAcc)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ViT-L_16")
    parser.add_argument("--epsMax", type=float, default=0.031)
    parser.add_argument("--num_clean_images", type=int, default=100)
    parser.add_argument("--queryLimit", type=int, default=3000)
    parser.add_argument("--numClasses", type=int, default=10)
    parser.add_argument("--attackSampleNum", type=int, default=1000)
    parser.add_argument("--batchSize", type=int, default=1)
    parser.add_argument("--imgSize", type=int, default=224)
    args =  parser.parse_args()
    
    RaySAttackVisionTransformer(args.model_name, 
                                args.numClasses, 
                                args.batchSize, 
                                args.imgSize, 
                                args.attackSampleNum, 
                                args.epsMax, 
                                args.queryLimit, 
                                args.num_clean_images)
