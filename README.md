# SymTax

To fix and streamline your workflow for running the model, generating taxonomy, and custom training, here is a
structured approach. Let's break down the instructions into coherent steps and ensure all paths, files, and commands are
clearly stated.

### running the SymTax models

python per.py

### Step 1: Running Baseline Datasets

To run the model on the arXiv, RefSeer, acl, or PeerRead datasets, follow these steps:

1. Open the `baseline_datasets.py` file.
2. Set the `enrich` variable to `True` or `False` based on whether you want to use the enricher.
3. Set the `dataset` variable to the desired dataset (e.g., `arxiv`, `refseer`, `peerread`, `acl`).

```python
# baseline_datasets.py

enrich = True  # or False
dataset = 'arxiv'  # or 'refseer', 'peerread'

```

4. Run the script to get the output metrics (recall, MRR, NCG):

```bash
python baseline_datasets.py
```

The `arxiv_acm_mapping_updated.csv` file should contain the mapping of the flat-level arXiv taxonomy to the tree-level
ACM taxonomy.

### Step 2: Custom Training

To custom train the model, follow these steps:

1. Edit the relevant arguments in `src/rerank/train.py`.

```python
# src/rerank/train.py

# Ensure the correct paths and parameters are set
# Example placeholder
train_data_path = 'path/to/train/data'
val_data_path = 'path/to/validation/data'
model_save_path = 'path/to/save/model'
# Add or modify other training parameters as needed
```

2. Modify the configuration file `src/rerank/config/arxiv/scibert/training_NN_prefetch.config` to reflect your desired
   settings.

```yaml
# src/rerank/config/arxiv/scibert/training_NN_prefetch.config

# Ensure the configuration parameters match your training requirements
batch_size: 32
learning_rate: 1e-5
num_epochs: 10
# Add or modify other configurations as needed
```

3. Navigate to the `src/rerank` directory and run the training script:

```bash
cd src/rerank
python train.py -config_file_path config/arxiv/scibert/training_NN_prefetch.config
```

### Summary of Commands

1. Running baseline datasets:

```bash
python baseline_datasets.py
```

2. Generating arXiv fusion taxonomy (in Jupyter Notebook):

```bash
# Open and run all cells in arxiv_acm_fusion.ipynb
```

3. Custom training:

```bash
cd src/rerank
python train.py -config_file_path config/arxiv/scibert/training_NN_prefetch.config
```

By following these steps, you can ensure your workflow is clear, and each part of the process is correctly configured
and executed.
