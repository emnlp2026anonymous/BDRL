import os
import json

from openai import AzureOpenAI


PROMPT_TEMPLATE = (
    "Role: You are an expert in molecular docking and drug discovery. "
    "You are a Pro-Filter, a sophisticated screening agent designed to filter, prioritize, and enrich "
    "candidate molecules with high binding potential to the target protein {protein_name} with the amino acid "
    "sequence: {protein_sequence}.\n\n"
    "Task: Identify the most promising {top_k} molecules in candidate SMILES {unique_samples}.\n\n"
    "Criteria:\n"
    "- Chemical Plausibility: ensure molecules are chemically valid and drug-like.\n"
    "- Target Compatibility: assess compatibility with the target using binding-relevant features.\n"
    "- Prioritize Molecules Similar to Memory: prioritize candidates structurally consistent with the top-30 "
    "  high-scoring molecules in memory {memory_top30}, but avoid selecting any SMILES already present in memory.\n"
    "- Knowledge-Informed Exploration: if selected candidates are insufficient, leverage chemical expertise and "
    "  memory to propose additional candidates with higher binding potential.\n"
    "- Filtering Count: output exactly {top_k} molecules.\n\n"
    "Output: Provide your output strictly as a JSON array of SMILES strings, with no explanations or additional text."
)

def samples_filter(samples, memory, top_k=64):
    protein_name = "1SYH"
    protein_seq = "GANKTVVVTTILESPYVMMKKNHEMLEGNERYEGYCVDLAAEIAKHCGFKYKLTIVGDGKYGARDADTKIWNGMVGELVYGKADIAIAPLTITLVREEVIDFSKPFMSLGISIMIKKGTPIESAEDLSKQTEIAYGTLDSGSTKEFFRRSKIAVFDKMWTYMRSAEPSVFVRTTAEGVARVRKSKGKYAYLLESTMNEYIEQRKPCDTMKVGGNLDSKGYGIATPKGSSLGNAVNLAVLKLNEQGLLDKLKNKWWYDKGECGS"

    # Filter out samples that are already in memory
    memory_smiles_set = set(memory["smiles"])
    unique_samples = [smi for smi in samples if smi not in memory_smiles_set]
    memory_top30 = memory.sort_values(by="scores", ascending=False).head(min(30,len(memory)))
    
    memory_top30 = {row["smiles"]: row["scores"] for _, row in memory_top30.iterrows()}
    
    endpoint = os.getenv("ENDPOINT_URL")
    deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not endpoint or not subscription_key:
        return unique_samples[:top_k]

    # Initialize Azure OpenAI client with key-based authentication
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2025-01-01-preview",
    )

    messages = [
        {"role": "system", "content": "You output only JSON arrays of SMILES."},
        {"role": "user", "content": PROMPT_TEMPLATE.format(
            unique_samples="" if not unique_samples else "from the following list:\n" + "\n".join(unique_samples),
            top_k=top_k,
            protein_name=protein_name,
            protein_sequence=protein_seq,
            memory_top30=json.dumps(memory_top30)
        ) + "\n"  }
    ]
    # print(messages)
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
        # Fallback to score-based top-k if LLM call fails or parsing fails
        candidates = unique_samples[:top_k]

    if len(candidates) > top_k:
        candidates = candidates[:top_k]
    if len(candidates) < top_k:
        remaining = [smi for smi in unique_samples if smi not in candidates]
        candidates.extend(remaining[: top_k - len(candidates)])

    return candidates
