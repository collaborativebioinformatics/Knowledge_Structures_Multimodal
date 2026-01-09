# Vertical Federated Learning: Image + RNA Example using NVFLARE

This repository demonstrates a **Vertical Federated Learning (VFL)** workflow using **NVFLARE**, where **image features** and **RNA-seq data** are kept private on separate clients and a **server fuses embeddings** for binary classification.  

The project uses **dummy data** for testing and is intended as a template for VFL experiments.

---

## Folder Structure

```bash
vfl_image_rna/
├── job.yaml
├── src/
│ └── vfl_executor.py
│ └── vfl_controller.py
└── config/
├── server.json
├── client_img.json
└── client_rna.json
```



---

## File Descriptions

## ***Job***
### **config/server.json**
- Server settings and executor path.
- Handles embedding fusion and gradient computation.

### **config/client_img.json / config/client_rna.json**
- Client settings and executor path.
- Handles forward pass (embedding generation) for image or RNA data.



### **src/vfl_executor.py, vfl_controller**
- Contains Python classes for ImageExecutor, RNAExecutor, and ServerExecutor.
- Implements forward and backward passes for the VFL workflow.

---

## Setup Instructions

1. Create a Python virtual environment:

```bash
python3 -m venv nvflare_vfl
source nvflare_vfl/bin/activate
```


2. Install nvflare:
   ```bash
   pip install nvflare
   ```

3. Verify folder structure (from project root):
```bash
ls vfl_image_rna/
ls vfl_image_rna/config/
```

4. Run the simulator
```bash
nvflare simulator ./job -n 2 -t 2
```

---

##  Customization

1. Change number of rounds: Edit num_rounds in recipe.yaml.
2. Change tasks: Update "tasks" in client/server JSONs.
3. Add more clients: Update deploy_map in job.yaml and add JSON files in config/.

---

##  WorkFlow Proof of Concept

This training flow is writting outside of nvflare in "Multimodal_Vertical_Federated_Learning_Attention_Based_Supervised.ipynb" and "Multimodal_Vertical_Federated_Learning_gated_fusion_Supervised.ipynb"







