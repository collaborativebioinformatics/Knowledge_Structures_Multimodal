import os 

import torch 

from clients.datasets import ClinicalDataset, RNADataset, clinical_collate_fn

# ----------------------------------------------------------------------- #
# Load Evaluation Data (only b/c we are simulating FL)
# ----------------------------------------------------------------------- #
def load_eval_data(dataset_path):
    IDS = ['3A_137', '3A_138', '3A_139', '3A_140', '3A_141', '3A_142', '3A_143', '3A_144', '3A_145', '3A_146', '3A_147', '3A_148', '3A_149', '3A_153', '3A_154', '3A_157', '3A_158', '3A_160', '3A_162', '3A_163', '3A_165', '3A_168', '3A_169', '3A_186', '3A_190', '3A_191']
    assert len(IDS) == 26, f"expected 26 IDS for evaluation (CHIMERA Task 3 Cohort A subset), got {len(IDS)}"
    clinical_files = [
        os.path.join(dataset_path, f"{id}/{id}_CD.json") for id in IDS
    ]  # Add all clinical file paths here
    clinical_ds = ClinicalDataset(clinical_files)

    rna_files = [
        os.path.join(dataset_path, f"{id}/{id}_RNA.json") for id in IDS
    ]  # Add all RNA file paths here
    rna_ds = RNADataset(rna_files)

    CDloader = torch.utils.data.DataLoader(
        clinical_ds, 
        batch_size=1, 
        shuffle=True,
        collate_fn=clinical_collate_fn
    )

    RNAloader = torch.utils.data.DataLoader(
        rna_ds, 
        batch_size=1, 
        shuffle=True
    )

    return CDloader, RNAloader

def evaluate(net, CDloader, RNAloader, device):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in data_loader:
            # (optional) use GPU to speed things up
            clinical, rnaseq, labels = data.to(device)
            outputs = net(clinical, rnaseq)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the network on the 26 test subjects from Cohort A: {100 * correct // total} %")
    return 100 * correct // total
    

