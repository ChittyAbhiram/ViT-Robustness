
import torch
import numpy
import ShuffleDefense
from ModelPlus import ModelPlus
import DataManagerPytorch as DMP
import AttackWrappersRayS
import AttackWrappersAdaptiveBlackBox
import AttackWrappersSAGA
from TransformerModels import VisionTransformer, CONFIGS
import BigTransferModels
from collections import OrderedDict
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os


def LoadViTLAndCIFAR10():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    numClasses = 10
    imgSize = 224
    batchSize = 8
    valLoader = DMP.GetCIFAR10Validation(imgSize, batchSize)
    config = CONFIGS["ViT-L_16"]
    model = VisionTransformer(config, imgSize, zero_head=True, num_classes=numClasses)
    dir = "ViT-L_16,cifar10,run0_15K_checkpoint.bin"
    dict = torch.load(dir, map_location=device)
    model.load_state_dict(dict)
    model.eval()
    modelPlus = ModelPlus("ViT-L_16", model, device, imgSizeH=imgSize, imgSizeW=imgSize, batchSize=batchSize)
    return valLoader, modelPlus



class ModelPlus:
    def __init__(self, name, model, device, imgSizeH=224, imgSizeW=224, batchSize=8):
        self.name = name
        self.model = model
        self.device = device
        self.model.to(device)
        self.imgSizeH = imgSizeH
        self.imgSizeW = imgSizeW
        self.batchSize = batchSize
    
    def validateD(self, dataLoader):
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for data in dataLoader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total




def SaveAdversarialImages(advLoader, clean_loader, filename_prefix, num_images=10):
    save_dir = "adversarial_images"
    os.makedirs(save_dir, exist_ok=True)
    adv_images = next(iter(advLoader))[0][:num_images]
    clean_images = next(iter(clean_loader))[0][:num_images]
    adv_grid = vutils.make_grid(adv_images, nrow=5, normalize=True)
    clean_grid = vutils.make_grid(clean_images, nrow=5, normalize=True)
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(clean_grid.permute(1, 2, 0).cpu())
    plt.title("Clean Images")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(adv_grid.permute(1, 2, 0).cpu())
    plt.title(f"Adversarial Images (ε={filename_prefix})")
    plt.axis('off')
    plt.savefig(f"{save_dir}/{filename_prefix}_comparison.png")
    plt.close()
    
    for i in range(num_images):
        adv_img = adv_images[i]
        clean_img = clean_images[i]
        
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(clean_img.permute(1, 2, 0).cpu())
        plt.title("Clean Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(adv_img.permute(1, 2, 0).cpu())
        plt.title(f"Adversarial Image (ε={filename_prefix})")
        plt.axis('off')
        
        plt.savefig(f"{save_dir}/{filename_prefix}_image_{i}.png")
        plt.close()
    
    return adv_images, clean_images


def CheckCleanAccuracy():
    print("Checking clean accuracy on CIFAR-10 validation dataset...")
    valLoader, defense = LoadViTLAndCIFAR10()
    cleanAcc = defense.validateD(valLoader)
    print("Clean accuracy:", cleanAcc)
    return cleanAcc



def RaySAttackVisionTransformer(epsMax):
    print(f"\nRunning RayS attack with epsMax={epsMax}...")
    valLoader, defense = LoadViTLAndCIFAR10()
    numClasses = 10
    attackSampleNum = 100  
    cleanLoader = DMP.GetCorrectlyIdentifiedSamplesBalancedDefense(defense, attackSampleNum, valLoader, numClasses) 
    queryLimit = 10000
    advLoader = AttackWrappersRayS.RaySAttack(defense, epsMax, queryLimit, cleanLoader) 
    robustAcc = defense.validateD(advLoader)
    cleanAcc = defense.validateD(valLoader)
    print("Epsilon:", epsMax)
    print("Queries used:", queryLimit)
    print("Robust accuracy:", robustAcc)
    print("Clean accuracy:", cleanAcc)
    SaveAdversarialImages(advLoader, cleanLoader, f"eps{str(epsMax).replace('.', '')}")
    
    return robustAcc, cleanAcc, advLoader


def RunExperimentAndWriteResults():
    print("=== Vision Transformer Robustness Experiment ===")
    print("Date: Saturday, March 01, 2025, 9:46 AM EST")
    clean_acc = CheckCleanAccuracy()
    robust_acc_031, _, _ = RaySAttackVisionTransformer(0.031)
    robust_acc_062, _, _ = RaySAttackVisionTransformer(0.062)
    print("\n=== Experimental Results ===")
    print("Model: ViT-L-16 on CIFAR-10")
    print("Attack: RayS (Black-box)")
    print("Number of samples: 100 (class-balanced)")
    print("Clean accuracy: {:.2f}%".format(clean_acc * 100))
    print("Robust accuracy (ε=0.031): {:.2f}%".format(robust_acc_031 * 100))
    print("Robust accuracy (ε=0.062): {:.2f}%".format(robust_acc_062 * 100))
    print("Accuracy drop (ε=0.031): {:.2f}%".format((clean_acc - robust_acc_031) * 100))
    print("Accuracy drop (ε=0.062): {:.2f}%".format((clean_acc - robust_acc_062) * 100))
    print("\nAdversarial images have been saved in the 'adversarial_images' directory.")
    

if __name__ == "__main__":
    RunExperimentAndWriteResults()












# def LoadShuffleDefenseAndCIFAR10(vis=False):
#     modelPlusList = []
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     numClasses = 10
#     imgSize = 224
#     batchSize = 8
#     valLoader = DMP.GetCIFAR10Validation(imgSize, batchSize)
#     config = CONFIGS["ViT-L_16"]
#     model = VisionTransformer(config, imgSize, zero_head=True, num_classes=numClasses, vis = vis)
#     dir = "ViT-L_16,cifar10,run0_15K_checkpoint.bin"
#     dict = torch.load(dir,  map_location=device)
#     model.load_state_dict(dict)
#     model.eval()
#     modelPlusV = ModelPlus("ViT-L_16", model, device, imgSizeH=imgSize, imgSizeW=imgSize, batchSize=batchSize)
#     modelPlusList.append(modelPlusV)
#     dirB = "BiT-M-R101x3-Run0.tar"
#     modelB = BigTransferModels.KNOWN_MODELS["BiT-M-R101x3"](head_size=numClasses, zero_head=False)
#     checkpoint = torch.load(dirB, map_location=device)
#     new_state_dict = OrderedDict()
#     for k, v in checkpoint["model"].items():
#         name = k[7:] 
#         new_state_dict[name] = v
    
#     modelB.load_state_dict(new_state_dict)
#     modelB.eval()
    
#     modelBig101Plus = ModelPlus("BiT-M-R101x3", modelB, device, imgSizeH=160, imgSizeW=128, batchSize=batchSize)
#     modelPlusList.append(modelBig101Plus)
    
#     defense = ShuffleDefense.ShuffleDefense(modelPlusList, numClasses)
#     return valLoader, defense




# # Method to do the RayS attack on a single Vision Transformer with epsMax=0.031
# def RaySAttackVisionTransformerEps031():
#     print("\nRunning RayS attack with epsMax=0.031...")
#     #Load the model and dataset
#     valLoader, defense = LoadViTLAndCIFAR10()
#     #Get the clean samples
#     numClasses = 10
#     attackSampleNum = 100  # Using 100 samples as required
#     cleanLoader = DMP.GetCorrectlyIdentifiedSamplesBalancedDefense(defense, attackSampleNum, valLoader, numClasses)
#     #Set the attack parameters 
#     epsMax = 0.031
#     queryLimit = 10000
#     #The next line does the actual attack on the defense 
#     advLoader = AttackWrappersRayS.RaySAttack(defense, epsMax, queryLimit, cleanLoader)
#     #Check the results 
#     robustAcc = defense.validateD(advLoader)
#     cleanAcc = defense.validateD(valLoader)
#     #Print the results 
#     print("Epsilon:", epsMax)
#     print("Queries used:", queryLimit)
#     print("Robust accuracy:", robustAcc)
#     print("Clean accuracy:", cleanAcc)
    
#     # Save the adversarial images
#     SaveAdversarialImages(advLoader, cleanLoader, "eps031")
    
#     return robustAcc, cleanAcc, advLoader

# # Method to do the RayS attack on a single Vision Transformer with epsMax=0.062
# def RaySAttackVisionTransformerEps062():
#     print("\nRunning RayS attack with epsMax=0.062...")
#     #Load the model and dataset
#     valLoader, defense = LoadViTLAndCIFAR10()
#     #Get the clean samples
#     numClasses = 10
#     attackSampleNum = 100  # Using 100 samples as required
#     cleanLoader = DMP.GetCorrectlyIdentifiedSamplesBalancedDefense(defense, attackSampleNum, valLoader, numClasses)
#     #Set the attack parameters 
#     epsMax = 0.062  # Higher epsilon value
#     queryLimit = 10000
#     #The next line does the actual attack on the defense 
#     advLoader = AttackWrappersRayS.RaySAttack(defense, epsMax, queryLimit, cleanLoader)
#     #Check the results 
#     robustAcc = defense.validateD(advLoader)
#     cleanAcc = defense.validateD(valLoader)
#     #Print the results 
#     print("Epsilon:", epsMax)
#     print("Queries used:", queryLimit)
#     print("Robust accuracy:", robustAcc)
#     print("Clean accuracy:", cleanAcc)
    
#     # Save the adversarial images
#     SaveAdversarialImages(advLoader, cleanLoader, "eps062")
    
#     return robustAcc, cleanAcc, advLoader



