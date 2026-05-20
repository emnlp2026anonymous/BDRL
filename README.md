# Fideli-Refiner Molecular Optimization

This repository contains a GPT-style SMILES generator for molecular design, with a two-stage workflow:

1. Pretrain a generative model on SMILES strings.
2. Fine-tune the model with reinforcement learning, LLM-based candidate filtering, low-fidelity molecular scoring, and optional Boltz-2 high-fidelity rescoring.

The default target used by the current scripts is `1SYH`, but the target-specific prompt and docking configuration can be changed in the corresponding files.

## Repository Structure

```text
.
+-- pretraining.py              # GPT pretraining on SMILES
+-- rl_finetuning.py            # RL fine-tuning loop
+-- model.py                    # GPT model definition
+-- dataset.py                  # SMILES dataset and collate logic
+-- vocabulary.py               # SMILES tokenization and vocabulary
+-- utils.py                    # sampling, likelihood, fingerprints, selection helpers
+-- scoring_function.py         # QED/SA/DRD2/GSK3B/JNK3/docking scoring
+-- scoring_function_boltz.py   # Boltz-2 scoring placeholder
+-- Pro_Filter.py               # LLM-based sample filtering agent
+-- Fideli_Refiner.py           # LLM-based Boltz rescoring selection agent
+-- data/
|   +-- ChEMBL_Smiles.csv
|   +-- vocab.txt
+-- docking/
|   +-- qvina02
|   +-- targets/
+-- oracle/
```

## Stage 1: Model Pretraining

Run pretraining on the ChEMBL SMILES dataset:

```bash
python pretraining.py \
  --run_name large_cont \
  --dataset chembl \
  --vocab_path data/vocab.txt \
  --ckpt_save_path ckpt/ \
  --num_epochs 20 \
  --batch_size 1024 \
  --max_length 128
```

The final checkpoint will be saved to:

```text
ckpt/large_cont/final.pt
```


## Stage 2: RL Fine-Tuning

After pretraining, run reinforcement learning fine-tuning:

```bash
python rl_finetuning.py \
  --run_name exp1 \
  --oracle docking_1SYH \
  --prior_path ckpt/large_cont/final.pt \
  --vocab_path data/vocab.txt \
  --n_steps 500 \
  --batch_size 128 \
  --memory_size 1000 \
  --replay 5
```


## Supported Low-Fidelity Oracles

`scoring_function.py` supports:

```text
QED
SA
DRD2
GSK3B
JNK3
JNK3_square
JNK3_half
docking_1SYH
docking_4LDE
docking_6Y2F
docking_PLPro_7JIR
docking_5R84
```

Docking modes require the corresponding target files under `docking/targets/` and the `qvina02` executable.

## LLM Agents

The project includes two LLM-based agents:

- `Pro_Filter.py`: filters sampled molecules before scoring.
- `Fideli_Refiner.py`: selects molecules that should receive high-fidelity Boltz-2 rescoring.

Both agents use Azure OpenAI through environment variables:

```bash
export ENDPOINT_URL="https://<your-resource-name>.openai.azure.com/"
export DEPLOYMENT_NAME="gpt-4o"
export AZURE_OPENAI_API_KEY="<your-api-key>"
```

If these variables are not set, the code falls back to simple score/order-based selection.

## Boltz-2 High-Fidelity Scoring

`scoring_function_boltz.py` is intentionally a placeholder. Boltz-2 is a large standalone project with its own installation, model weights, inputs, and runtime requirements, so it is not included or vendored in this repository. Users should install and configure Boltz-2 separately in their own environment, then connect that setup by implementing:

```python
def get_scores(smiles):
    ...
```

Expected input:

```python
list[str]
```

Expected output:

```python
dict[str, float]
```

where each key is a SMILES string and each value is a normalized score. Higher scores should be better because `rl_finetuning.py` uses Boltz scores to replace or update molecule scores.

Reference:

```bibtex
@article{passaro2025boltz2,
  author = {Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and Reveiz, Mateo and Thaler, Stephan and Somnath, Vignesh Ram and Getz, Noah and Portnoi, Tally and Roy, Julien and Stark, Hannes and Kwabi-Addo, David and Beaini, Dominique and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  year = {2025},
  doi = {10.1101/2025.06.14.659707},
  journal = {bioRxiv}
}
```

## Changing the Target Protein

The default target in the LLM prompts is `1SYH`. To use another target, update:

- `protein_name` and `protein_seq` in `Pro_Filter.py`
- `protein_name` and `protein_seq` in `Fideli_Refiner.py`
- docking target configuration in `scoring_function.py`, if using docking objectives
