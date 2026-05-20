
import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from rdkit import Chem, DataStructs
from Pro_Filter import samples_filter
from model import GPT, GPTConfig
from vocabulary import read_vocabulary
from utils import sample_SMILES, likelihood, to_tensor, calc_fingerprints,select_boltz
from scoring_function import get_scores, int_div
from scoring_function_boltz import get_scores as get_boltz_scores


torch.manual_seed(44)
np.random.seed(44)
random.seed(44)

def compute_boltz_scores(smiles):

    return get_boltz_scores(smiles)

def boltz_memory_update(boltz_memory, combined_smiles, boltz_scores_dict, seqs):
    """Update Boltz memory with scored SMILES and their token sequences."""
    for smi in combined_smiles:
        boltz_score = boltz_scores_dict.get(smi, None)
        if boltz_score is not None and boltz_score > 0:
            fp, smiles_i = calc_fingerprints([smi])
            seq = seqs.get(smi, None)
            if seq is not None:
                if isinstance(seq, np.ndarray):  # Check if seq is a numpy array
                    seq = torch.tensor(seq)  # Convert numpy array to tensor
                new_data = pd.DataFrame({
                    "smiles": smiles_i[0],
                    "scores": boltz_score,
                    "seqs": [seq.cpu().numpy()],
                    "fps": fp[0]
                })
                boltz_memory = pd.concat([boltz_memory, new_data], ignore_index=True, sort=False)

    boltz_memory = boltz_memory.drop_duplicates(subset=["smiles"])
    boltz_memory = boltz_memory.sort_values('scores', ascending=False).reset_index(drop=True)
    return boltz_memory

def memory_update(memory, smiles, scores, seqs, memory_size, replay):
    """Independent function for memory update."""
    scores = list(scores)
    seqs_list = [seqs[i, :].cpu().numpy() for i in range(len(smiles))]

    for i in range(len(smiles)):
        if scores[i] < 0:
            continue
        fp, smiles_i = calc_fingerprints([smiles[i]])
        new_data = pd.DataFrame({
            "smiles": smiles_i[0],
            "scores": scores[i],
            "seqs": [seqs_list[i]],
            "fps": fp[0]
        })
        memory = pd.concat([memory, new_data], ignore_index=True, sort=False)

    memory = memory.drop_duplicates(subset=["smiles"])
    memory = memory.sort_values('scores', ascending=False).reset_index(drop=True)
    if len(memory) > memory_size:
        memory = memory.head(memory_size)


    if len(memory) > replay:
        if replay > 0:
            s = min(len(memory), replay)
            experience = memory.head(5 * replay).sample(s)
            experience = experience.reset_index(drop=True)
            smiles += list(experience["smiles"])
            scores += list(experience["scores"])
            for index in experience.index:
                seqs = torch.cat(
                    (seqs, torch.tensor(experience.loc[index, "seqs"],
                                        dtype=torch.long).view(1, -1).cuda()), dim=0
                )

    return memory, smiles, np.array(scores), seqs

def select_highest_non_boltz(memory, boltz_memory, samples=None, scores=None):
    if memory.empty:
        if samples is not None and scores is not None:
            highest_idx = np.argmax(scores)
            return [samples[highest_idx]]
        return []

    boltz_smiles_set = set(boltz_memory["smiles"])
    non_boltz_memory = memory[~memory["smiles"].isin(boltz_smiles_set)]
    if non_boltz_memory.empty:
        return []
    highest_scoring_smiles = non_boltz_memory.iloc[0]["smiles"]  # Select the top-scoring SMILES
    return [highest_scoring_smiles]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="")
    parser.add_argument('--model_type', type=str, default="gpt")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--oracle', type=str, default="docking_1SYH")
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_embd', type=int, default=384)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--n_steps', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sigma', type=float, default=100)
    parser.add_argument('--kl_beta', type=float, default=0.001)
    parser.add_argument('--clip_eps', type=float, default=0.2)
    parser.add_argument('--policy_updates', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--memory_size', type=int, default=1000)
    parser.add_argument('--replay', type=int, default=5)
    parser.add_argument('--prior_path', type=str, default="ckpt/large_cont/final.pt")
    parser.add_argument('--vocab_path', type=str, default="data/vocab.txt")
    parser.add_argument('--output_dir', type=str, default="rl_log/")
    args = parser.parse_args()
    print(args)

    run_dir = args.output_dir + f"{args.oracle}_{args.run_name}/"
    writer = SummaryWriter(run_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    writer.add_text("configs", str(args))


    voc = read_vocabulary(args.vocab_path)


    model_config = GPTConfig(
        voc.__len__(),
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.max_length
    )


    if args.model_type == "gpt":
        prior = GPT(model_config).to(args.device)
        agent = GPT(model_config).to(args.device)
        optimizer = agent.configure_optimizers(
            weight_decay=0.1,
            learning_rate=args.learning_rate,
            betas=(0.9, 0.95)
        )


    prior.load_state_dict(torch.load(args.prior_path), strict=True)
    for param in prior.parameters():
        param.requires_grad = False
    prior.eval()


    agent.load_state_dict(torch.load(args.prior_path), strict=True)
    agent.eval()


    memory = pd.DataFrame(columns=["smiles", "scores", "seqs", "fps"])
    boltz_memory = pd.DataFrame(columns=["smiles", "scores", "seqs", "fps"])
    if not os.path.exists(f'rl_outputs/{args.oracle}_{args.run_name}/'):
        os.makedirs(f'rl_outputs/{args.oracle}_{args.run_name}/')
    if not os.path.exists(f'rl_ckpts/{args.oracle}_{args.run_name}/'):
        os.makedirs(f'rl_ckpts/{args.oracle}_{args.run_name}/')


    boltz_scores_log_path = os.path.join("logs", run_dir, "boltz_scores_log.txt")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(boltz_scores_log_path), exist_ok=True)


    for step in tqdm(range(args.n_steps)):
        print('Step:',step+1)
        if step < 0:
            samples, seqs, entropies = sample_SMILES(agent, voc, n_mols=args.batch_size, temperature=args.temperature)
        else:

            samples, seqs, entropies = sample_SMILES(agent, voc, n_mols=args.batch_size, temperature=args.temperature)
            # Filter samples after sampling
            filtered_samples = samples_filter(samples, memory, top_k=64)

            replacement_seqs = [voc.encode(smi) for smi in filtered_samples]  # Encode each SMILES string
            replacement_tensors = [torch.tensor(seq, dtype=torch.long).cuda() for seq in replacement_seqs]
            # Pad all tensors to the same length as seqs and stack them into a batch
            max_len = seqs.size(1)
            padded_tensors = []
            for tensor in replacement_tensors:
                if tensor.size(0) < max_len:
                    padded = torch.zeros(max_len, dtype=torch.long).cuda()
                    padded[:tensor.size(0)] = tensor
                    padded_tensors.append(padded)
                else:
                    padded_tensors.append(tensor[:max_len])  # Truncate if longer than max_len

            seqs = torch.stack(padded_tensors, dim=0).to("cuda")  # Stack into a batch
            seqs = torch.clamp(seqs, min=0,max=106)
            entropies = entropies[:len(filtered_samples)]
            samples = filtered_samples


        scores = np.array(get_scores(samples, mode=args.oracle))


        do_boltz = ((step + 1) % 5 == 0)
        if do_boltz:
            print("Using Boltz scoring based on memory:", step)

            memory_top100 = memory.head(100)
            boltz_smiles = select_boltz(list(memory_top100["smiles"]), list(memory_top100["scores"]), boltz_memory)
            print(f"Selected {len(boltz_smiles)} samples from memory for Boltz scoring.")

            if len(boltz_smiles) > 0:
                print("Computing Boltz scores...")
                boltz_scores_dict = compute_boltz_scores(boltz_smiles)
                with open(boltz_scores_log_path, "a", encoding="utf-8") as log_file:
                    for smi in boltz_smiles:
                        original_score = memory.loc[memory["smiles"] == smi, "scores"].values[0]
                        boltz_score = boltz_scores_dict.get(smi, None)
                        log_file.write(json.dumps({
                            "step": int(step + 1),
                            "smiles": smi,
                            "original_score": original_score,
                            "boltz_score": boltz_score
                        }) + "\n")


                        if boltz_score is not None and boltz_score > 0:
                            memory.loc[memory["smiles"] == smi, "scores"] = boltz_score if boltz_score > 0 else original_score


                boltz_memory = boltz_memory_update(
                    boltz_memory, boltz_smiles, boltz_scores_dict,
                    {smi: seq for smi, seq in zip(memory_top100["smiles"], memory_top100["seqs"])}
                )
        else:
            boltz_mask = np.zeros(len(samples), dtype=bool)

        writer.add_scalar('Entropy', np.mean(entropies.detach().cpu().numpy()), step)
        writer.add_scalar('Step Mean', np.mean(np.array(scores)), step)
        if (step + 1) % 10 == 0:
            smiles_df = pd.DataFrame(samples, columns=["SMILES"])
            smiles_df.to_csv(f'rl_outputs/{args.oracle}_{args.run_name}/smiles_step{step+1}.csv', index=False)
            writer.add_scalar('Step Div', int_div(samples), step)
            torch.save(agent.state_dict(), f'rl_ckpts/{args.oracle}_{args.run_name}/agent_step{step+1}.pt')

        memory, samples, scores, seqs = memory_update(
            memory, samples, scores, seqs, args.memory_size, args.replay
        )

        with torch.no_grad():
            old_likelihood = likelihood(agent, seqs).detach()
            prior_likelihood = likelihood(prior, seqs).detach()
        rewards = to_tensor(np.array(scores)).float()
        advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)
        advantages = advantages.detach()

        for _ in range(args.policy_updates):
            agent_likelihood = likelihood(agent, seqs)

            ratio = torch.exp(old_likelihood - agent_likelihood)
            clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
            policy_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()

            log_ref_over_policy = agent_likelihood - prior_likelihood
            kl_loss = (torch.exp(log_ref_over_policy) - 1.0 - log_ref_over_policy).mean()
            loss = policy_loss + args.kl_beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        writer.add_scalar('Memory Mean', np.mean(np.array(memory["scores"])), step)
        writer.add_scalar('Prior Likelihood', np.mean(prior_likelihood.detach().cpu().numpy()), step)
        writer.add_scalar('Agent Likelihood', np.mean(agent_likelihood.detach().cpu().numpy()), step)
        writer.add_scalar('GRPO Policy Loss', policy_loss.detach().cpu().item(), step)
        writer.add_scalar('KL Loss', kl_loss.detach().cpu().item(), step)
        writer.add_scalar('Total Loss', loss.detach().cpu().item(), step)

        writer.add_scalar('Top-1', memory["scores"][0], step)
        writer.add_scalar('Top-10 Mean', np.mean(np.array(memory["scores"][:10])), step)
        writer.add_scalar('Top-100 Mean', np.mean(np.array(memory["scores"][:100])), step)

        if (step + 1) % 10 == 0:
            writer.add_scalar('Top-100 Div', int_div(list(memory["smiles"][:100])), step)

        if (step + 1) % 100 == 0:
            memory.to_csv(f'rl_outputs/{args.oracle}_{args.run_name}/memory_step{step+1}.csv')
            boltz_memory.to_csv(f'rl_outputs/{args.oracle}_{args.run_name}/boltz_memory_step{step+1}.csv')



    memory.to_csv(f'rl_outputs/{args.oracle}_{args.run_name}/final_{args.n_steps}steps.csv')
    boltz_memory.to_csv(f'rl_outputs/{args.oracle}_{args.run_name}/final_boltz_memory_{args.n_steps}steps.csv')
    torch.save(agent.state_dict(), f'rl_ckpts/{args.oracle}_{args.run_name}_finalagent.pt')


    print(f'top-1 score: {memory["scores"][0]}')
    print(f'top-10 score: {np.mean(np.array(memory["scores"][:10]))}')
    print(f'top-100 score: {np.mean(np.array(memory["scores"][:100]))}, diversity: {int_div(list(memory["smiles"][:100]))}')

    writer.close()
