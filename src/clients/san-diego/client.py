"""
Adapted from NVFlare "Hello PyTorch" (hello-pt) example code.
"""
import os 

import torch 
from torch import nn 

import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter

from server.model import SimpleNetwork
from clients.datasets import ClinicalDataset, RNADataset

def main():
    DATASET_PATH="/Users/tyleryang/Developer/CMU-NVIDIA-Hackathon/rna-cd-data/"
    # ----------------------------------------------------------------------- #
    # Client-Specific Constants
    # ----------------------------------------------------------------------- #
    IDS = ['3A_001', '3A_002', '3A_003', '3A_004', '3A_005', '3A_006', '3A_007', '3A_008', '3A_009', '3A_010', '3A_011', '3A_012', '3A_013', '3A_014', '3A_015', '3A_016', '3A_017', '3A_018', '3A_019', '3A_020', '3A_021', '3A_022', '3A_023', '3A_024', '3A_025', '3A_026', '3A_027', '3A_028', '3A_029', '3A_030', '3A_031', '3A_033', '3A_034', '3A_035', '3A_036', '3A_037', '3A_038', '3A_039', '3A_040', '3A_041', '3A_042', '3A_043', '3A_044', '3A_045', '3A_046', '3A_047', '3A_049', '3A_050', '3A_052', '3A_053', '3A_055', '3A_056', '3A_057', '3A_058', '3A_059', '3A_060', '3A_061', '3A_062', '3A_063', '3A_064', '3A_066', '3A_067', '3A_068', '3A_070', '3A_071', '3A_072', '3A_073', '3A_074', '3A_075', '3A_076', '3A_077', '3A_087', '3A_088', '3A_089', '3A_091', '3A_092', '3A_093', '3A_094', '3A_095', '3A_097', '3A_098', '3A_100', '3A_105', '3A_108', '3A_110', '3A_111', '3A_113', '3A_114', '3A_115', '3A_116', '3A_123', '3A_124', '3A_125', '3A_126', '3A_127', '3A_129', '3A_130', '3A_134', '3A_135', '3A_136']
    assert len(IDS) == 100, f"expected 100 IDS in San Diego, got {len(IDS)}"

    # ----------------------------------------------------------------------- #
    # Initialize Model
    # ----------------------------------------------------------------------- #

    # Shared across all clients
    model = SimpleNetwork(rna_dim=19359, clinical_dim=13)

    # Free for Client to modify
    config = {
        "batch_size": 10,
        "epochs": 2,
        "learning_rate": 1e-2
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)

    # ----------------------------------------------------------------------- #
    # Load Client-Specific Data for Federated Training
    # ----------------------------------------------------------------------- #
    clinical_files = [
        os.path.join(DATASET_PATH, f"{id}/{id}_CD.json") for id in IDS
    ]  # Add all clinical file paths here
    clinical_ds = ClinicalDataset(clinical_files)

    rna_files = [
        os.path.join(DATASET_PATH, f"{id}/{id}_RNA.json") for id in IDS
    ]  # Add all RNA file paths here
    rna_ds = RNADataset(rna_files)

    CDloader = torch.utils.data.DataLoader(
        clinical_ds, 
        batch_size=config['batch_size'], 
        shuffle=True
    )

    RNAloader = torch.utils.data.DataLoader(
        rna_ds, 
        batch_size=config['batch_size'], 
        shuffle=True
    )

    # ----------------------------------------------------------------------- #
    # Load Evaluation Data (only b/c we are simulating FL)
    # ----------------------------------------------------------------------- #
    def evaluate(net, data_loader, device):
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
    
    TEST_IDS = ['3A_137', '3A_138', '3A_139', '3A_140', '3A_141', '3A_142', '3A_143', '3A_144', '3A_145', '3A_146', '3A_147', '3A_148', '3A_149', '3A_153', '3A_154', '3A_157', '3A_158', '3A_160', '3A_162', '3A_163', '3A_165', '3A_168', '3A_169', '3A_186', '3A_190', '3A_191']
    assert len(TEST_IDS) == 26, f"expected 26 IDS for evaluation, got {len(TEST_IDS)}"


    # ----------------------------------------------------------------------- #
    # NVFlare Initialization
    # ----------------------------------------------------------------------- #
    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]
    summary_writer = SummaryWriter()
    print(f"San Diego Hospital corresponds to site_name: {client_name}")

    # ----------------------------------------------------------------------- #
    # NVFlare Training Loop
    # ----------------------------------------------------------------------- #
    while flare.is_running():
        global_model = flare.receive()
        print(f"San Diego Hospital site_name: {client_name}")
        print(f"current_round={global_model.current_round}") # type: ignore

        model.load_state_dict(global_model.params) # type: ignore
        model.to(device)

        # TODO: evaluate on received model
        accuracy = evaluate(model, test_loader, device)

        steps = config['epochs'] * len(train_loader)
        for epoch in range(config['epochs']):
            running_loss = 0.0
            for i, batch in enumerate(train_loader):
                images, labels = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()

                predictions = model(images)
                cost = loss(predictions, labels)
                cost.backward()
                optimizer.step()

                running_loss += cost.item()
                if i % 5 == 4:
                    avg_loss = running_loss / 5
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}")

                    # Optional: Log metrics
                    global_step = global_model.current_round * steps + epoch * len(train_loader) + i # type: ignore
                    summary_writer.add_scalar(tag="loss", scalar=avg_loss, global_step=global_step)

                    print(f"site={client_name}, Epoch: {epoch}/{config['epochs']}, Iteration: {i}, Loss: {running_loss}")
                    running_loss = 0.0

        print(f"Finished Training for San Diego Hospital, site_name: {client_name}")

        PATH = "./san-diego.pth"
        torch.save(model.state_dict(), PATH)

        # (7) construct trained FL model
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            metrics={"accuracy": accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        print(f"site: {client_name}, sending model to server.")
        # (8) send model back to NVFlare
        flare.send(output_model)

if __name__ == "__main__":
    main()