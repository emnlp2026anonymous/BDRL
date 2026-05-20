import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
from tdc import Evaluator

from Fideli_Refiner import llm_select_boltz
from vocabulary import SMILESTokenizer, read_vocabulary


def randomize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    ans = list(range(mol.GetNumAtoms()))
    if mol is None or ans == []:
        return smiles
    np.random.shuffle(ans)
    new_mol = Chem.RenumberAtoms(mol, ans)
    return Chem.MolToSmiles(new_mol, canonical=False)


def likelihood(model, seqs):
    nll_loss = nn.NLLLoss(reduction="none", ignore_index=0)
    seqs = seqs.cuda()
    logits, _ = model(seqs[:, :-1])
    log_probs = logits.log_softmax(dim=2)
    loss_per_token = nll_loss(log_probs.transpose(1, 2), seqs[:, 1:])
    return loss_per_token.sum(dim=1)


@torch.no_grad()
def sample_SMILES(model, voc, n_mols=100, block_size=100, temperature=1.0, top_k=10, average_entropy=True):
    codes = torch.zeros((n_mols, 1), dtype=torch.long).to("cuda")
    codes[:] = voc["^"]
    entropies = torch.zeros(n_mols).to("cuda")
    valid_counts = torch.zeros(n_mols).to("cuda")

    model.eval()
    for _ in range(block_size - 1):
        logits, _ = model(codes)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            logits = top_k_logits(logits, k=top_k)

        probs = logits.softmax(dim=-1)
        code_i = torch.multinomial(probs, num_samples=1)
        codes = torch.cat((codes, code_i), dim=1)

        step_entropy = torch.special.entr(probs).sum(dim=1)
        mask = (code_i.view(-1) != 0).float()
        entropies += step_entropy * mask
        valid_counts += mask

        if code_i.sum() == 0:
            break

    if average_entropy:
        entropies = entropies / valid_counts.clamp(min=1)

    smiles = []
    tokenizer = SMILESTokenizer()
    for i in range(n_mols):
        tokens_i = voc.decode(np.array(codes[i, :].cpu()))
        smiles_i = tokenizer.untokenize(tokens_i)
        smiles.append(smiles_i)

    return smiles, codes, entropies


def model_validity(model, vocab_path, n_mols=100, block_size=100):
    evaluator = Evaluator(name="Validity")
    voc = read_vocabulary(vocab_path)
    smiles, _, _ = sample_SMILES(model, voc=voc, n_mols=n_mols, block_size=block_size, top_k=10)
    return evaluator(smiles)


def calc_fingerprints(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, radius=2, nBits=2048) for x in mols]
    smiles_canonicalized = [Chem.MolToSmiles(x, isomericSmiles=False) for x in mols]
    return fps, smiles_canonicalized


def top_k_logits(logits, k):
    v, _ = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float("Inf")
    return out


def to_tensor(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)


def select_boltz(smiles, scores, boltz_memory):
    """Select Boltz re-scoring candidates entirely with the LLM agent."""
    existing = set(boltz_memory["smiles"]) if len(boltz_memory) > 0 else set()
    candidates = {
        smi: float(score)
        for smi, score in zip(smiles, scores)
        if smi not in existing
    }
    if not candidates:
        return []

    return llm_select_boltz(candidates, boltz_memory)
