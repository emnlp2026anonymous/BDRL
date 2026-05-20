import os
import json
from typing import Dict, List, Optional

from openai import AzureOpenAI


PROMPT_TEMPLATE = (
    "Role: You are an expert in molecular docking and drug discovery. "
    "You are a Fideli-Refiner, an advanced agent designed to selectively allocate high-fidelity "
    "evaluations to candidate molecules with potentially inaccurate low-fidelity binding affinities "
    "to the target protein {protein_name} with the amino acid sequence: {protein_sequence}.\n\n"
    "Task: Prioritize and identify the most inaccurate molecules requiring re-evaluation. "
    "Select a minimum of {min_k} and a maximum of {max_k} molecules for re-scoring.\n\n"
    "Context:\n"
    "- Candidate SMILES strings are provided as Experience Memory with low-fidelity scores: {experience_memory}\n"
    "- Boltz memory includes previously evaluated molecules and their high-fidelity scores: {boltz_memory}\n"
    "- Training Progress: Total training consists of {total_steps} steps. Current step: {step}.\n\n"
    "Determine the number of molecules to re-score based on the current training stage:\n"
    "- Early stages (e.g., steps 0-100): allocate fewer evaluations due to high uncertainty.\n"
    "- Mid stages (e.g., steps 100-400): gradually increase evaluations as higher-quality candidates emerge.\n"
    "- Final stages (e.g., steps 400-500): reduce evaluations as most high-quality molecules have been evaluated.\n\n"
    "Selection criteria (prioritize in this order):\n"
    "1. High-potency candidates: prioritize molecules with high raw low-fidelity Vina scores.\n"
    "2. Error-prone structures: molecules structurally similar to those in Boltz memory that show large\n"
    "   low- vs high-fidelity discrepancies.\n"
    "3. Intrinsic scoring bias: use chemical knowledge to flag scaffolds likely to be mis-scored.\n\n"
    "Output: Return ONLY a valid JSON array of SMILES strings, no explanations or extra text."
)
def _build_messages(
    smiles_scores: Dict[str, float],
    boltz_memory,
    step: Optional[int],
    total_steps: Optional[int],
    min_k: int,
    max_k: int,
):
    ranked = sorted(smiles_scores.items(), key=lambda x: float(x[1]), reverse=True)
    boltz_memory = {entry["smiles"]: entry["scores"] for _, entry in boltz_memory.iterrows()}
    protein_name = "1SYH"
    protein_seq = "GANKTVVVTTILESPYVMMKKNHEMLEGNERYEGYCVDLAAEIAKHCGFKYKLTIVGDGKYGARDADTKIWNGMVGELVYGKADIAIAPLTITLVREEVIDFSKPFMSLGISIMIKKGTPIESAEDLSKQTEIAYGTLDSGSTKEFFRRSKIAVFDKMWTYMRSAEPSVFVRTTAEGVARVRKSKGKYAYLLESTMNEYIEQRKPCDTMKVGGNLDSKGYGIATPKGSSLGNAVNLAVLKLNEQGLLDKLKNKWWYDKGECGS"
    lines = [f"{i+1}. {smi} :: score={score:.4f}" for i, (smi, score) in enumerate(ranked)]
    experience_memory = "\n".join(lines)
    content = PROMPT_TEMPLATE.format(
        protein_name=protein_name,
        protein_sequence=protein_seq,
        experience_memory=experience_memory,
        boltz_memory=json.dumps(boltz_memory, ensure_ascii=True),
        total_steps=total_steps if total_steps is not None else "unknown",
        step=step if step is not None else "unknown",
        min_k=min_k,
        max_k=max_k,
    )
    return [
        {"role": "system", "content": "You output only JSON arrays of SMILES."},
        {"role": "user", "content": content},
    ]


def llm_select_boltz(
    smiles_scores: Dict[str, float],
    boltz_memory,
    step: Optional[int] = None,
    total_steps: Optional[int] = None,
    min_k: int = 3,
    max_k: int = 10,
) -> List[str]:
    """Call GPT to rank/select samples for Boltz scoring."""
    max_k = min(max_k, len(smiles_scores))
    min_k = min(min_k, max_k)
    if max_k <= 0 or not smiles_scores:
        return []

    endpoint = os.getenv("ENDPOINT_URL")
    deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not endpoint or not subscription_key:
        sorted_pairs = sorted(smiles_scores.items(), key=lambda x: float(x[1]), reverse=True)
        return [p[0] for p in sorted_pairs[:min_k]]

    # Initialize Azure OpenAI client with key-based authentication
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2025-01-01-preview",
    )

    messages = _build_messages(smiles_scores, boltz_memory, step, total_steps, min_k, max_k)

    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=0.2
        )
        raw = resp.choices[0].message.content.strip()
        data = json.loads(raw)
        candidates = data if isinstance(data, list) else []
    except Exception:
        # Fallback to score-based min_k if LLM call fails or parsing fails
        sorted_pairs = sorted(smiles_scores.items(), key=lambda x: float(x[1]), reverse=True)
        return [p[0] for p in sorted_pairs[:min_k]]

    picked = []
    seen = set()
    for smi in candidates:
        if smi in seen:
            continue
        if smi in smiles_scores:
            picked.append(smi)
            seen.add(smi)
        if len(picked) >= max_k:
            break

    if len(picked) < min_k:
        remaining_pairs = [p for p in sorted(smiles_scores.items(), key=lambda x: float(x[1]), reverse=True) if p[0] not in seen]
        picked.extend([p[0] for p in remaining_pairs[: min_k - len(picked)]])

    return picked
