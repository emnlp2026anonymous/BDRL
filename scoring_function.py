import os
import glob
import numpy as np
import uuid
import time

import subprocess
import multiprocessing

from rdkit.Chem import MolFromSmiles
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from openbabel import pybel
from tdc import Oracle, Evaluator


def int_div(smiles):
    evaluator = Evaluator(name = 'Diversity')
    return evaluator(smiles)


def get_scores(smiles, mode="QED", n_process=8):
    
    smiles_groups = []
    group_size = len(smiles) / n_process
    for i in range(n_process):
        smiles_groups += [smiles[int(i * group_size):int((i + 1) * group_size)]]

    temp_data = []
    pool = multiprocessing.Pool(processes = n_process)
    for index in range(n_process):
        temp_data.append(pool.apply_async(get_scores_subproc, args=(smiles_groups[index], mode, )))
    pool.close()
    pool.join()
    scores = []
    for index in range(n_process):
        scores += temp_data[index].get()

    for filename in glob.glob("docking/mols/*"):
        if os.path.exists(filename):
            os.remove(filename)

    return scores

def get_scores_subproc(smiles, mode):
    scores = []
    mols = [MolFromSmiles(s) for s in smiles]
    oracle_QED = Oracle(name='QED')
    oracle_SA = Oracle(name='SA')

    if mode == "QED":
        for i in range(len(smiles)):
            if mols[i] != None:
                scores += oracle_QED([smiles[i]])
            else:
                scores += [-1.0]

    elif mode == "SA":
        for i in range(len(smiles)):
            if mols[i] != None:
                scores += oracle_SA([smiles[i]])
            else:
                scores += [-1.0]

    elif mode == "DRD2":
        oracle = Oracle(name='DRD2')
        for i in range(len(smiles)):
            if mols[i] != None:
                scores += oracle([smiles[i]])
            else:
                scores += [-1.0]

    elif mode == "GSK3B":
        oracle = Oracle(name='GSK3B')
        for i in range(len(smiles)):
            if mols[i] != None:
                scores += oracle([smiles[i]])
            else:
                scores += [-1.0]

    elif mode == "JNK3":
        oracle = Oracle(name='JNK3')
        for i in range(len(smiles)):
            if mols[i] != None:
                scores += oracle([smiles[i]])
            else:
                scores += [-1.0]

    elif mode == "JNK3_square":
        oracle = Oracle(name='JNK3')
        for i in range(len(smiles)):
            if mols[i] != None:
                scores += [oracle([smiles[i]])[0] ** 2]
            else:
                scores += [-1.0]

    elif mode == "JNK3_half":
        oracle = Oracle(name='JNK3')
        for i in range(len(smiles)):
            if mols[i] != None:
                scores += [oracle([smiles[i]])[0] / 2]
            else:
                scores += [-1.0]

    elif mode == "docking_1SYH":
        for i in range(len(smiles)):
            if mols[i] != None:
                docking_score = docking(smiles[i], receptor_file="docking/targets/1syh.pdbqt", box_center=[21.49, 13.46, 23.18])
                scores += [reverse_sigmoid_transformation(docking_score)]
            else:
                scores += [-1.0]

    elif mode == "docking_4LDE":
        for i in range(len(smiles)):
            if mols[i] != None:
                
                docking_score = docking(smiles[i], receptor_file="docking/targets/4lde.pdbqt", box_center=[-2.94, -12.92, -50.99])
                scores += [reverse_sigmoid_transformation(docking_score)]
            else:
                scores += [-1.0]
  
    elif mode == "docking_6Y2F":
            for i in range(len(smiles)):
                if mols[i] != None:
                    
                    docking_score = docking(smiles[i], receptor_file="docking/targets/6y2f.pdbqt", box_center=[11.03, -0.61, 20.84])
                    scores += [reverse_sigmoid_transformation(docking_score)]
                else:
                    scores += [-1.0]

    elif mode == "docking_PLPro_7JIR":
        for i in range(len(smiles)):
            if mols[i] != None:
                docking_score = docking(smiles[i], receptor_file="data/targets/7jir+w2.pdbqt", box_center=[51.51, 32.16, -0.55])
                scores += [reverse_sigmoid_transformation(docking_score)]
            else:
                scores += [-1.0]
    elif mode == "docking_5R84":
        for i in range(len(smiles)):
            if mols[i] != None:
                docking_score = docking(smiles[i], receptor_file="data/targets/5r84.pdbqt", box_center=[-8, 6, 8])
                scores += [reverse_sigmoid_transformation(docking_score)]
            else:
                scores += [-1.0]    

    else:
        raise Exception("Scoring function undefined!")


    return scores


def docking(smiles, receptor_file, box_center, box_size=[20, 20, 20]):
    if smiles == "":
        return 100.0

    label = generate_unique_label()

    ligand_mol_file = f"./docking/mols/mol_{label}.mol"
    ligand_pdbqt_file = f"./docking/mols/mol_{label}.pdbqt"
    docking_pdbqt_file = f"./docking/mols/dock_{label}.pdbqt"

    # 3D conformation of SMILES
    try:
        run_line = 'obabel -:%s --gen3D -O %s' % (smiles, ligand_mol_file)
        result = subprocess.check_output(run_line.split(), stderr=subprocess.STDOUT,
                    timeout=10, universal_newlines=True)
        # print('successfully generated 3D conformation for SMILES: %s' % smiles)
    except Exception as e:
        print(f"3D Generation Failed: {smiles}")
        # print(e)
        return 100.0

    # docking by quick vina
    try:
        ms = list(pybel.readfile("mol", ligand_mol_file))
        m = ms[0]
        m.write("pdbqt", ligand_pdbqt_file, overwrite=True)
        run_line = 'docking/qvina02 --receptor %s --ligand %s --out %s' % (receptor_file, ligand_pdbqt_file, docking_pdbqt_file)
        run_line += ' --center_x %s --center_y %s --center_z %s' % (box_center[0], box_center[1], box_center[2])
        run_line += ' --size_x %s --size_y %s --size_z %s' % (box_size[0], box_size[1], box_size[2])
        run_line += ' --cpu %d' % (8)
        run_line += ' --exhaustiveness %d ' % (4)
        result = subprocess.check_output(run_line.split(),
                                            stderr=subprocess.STDOUT,
                                            timeout=100, universal_newlines=True)
        result_lines = result.split('\n')
        affinity_list = list()
        check_result = False
        for result_line in result_lines:
            if result_line.startswith('-----+'):
                check_result = True
                continue
            if not check_result:
                continue
            if result_line.startswith('Writing output'):
                break
            if result_line.startswith('Refine time'):
                break
            lis = result_line.strip().split()
            if not lis[0].isdigit():
                break
            affinity = float(lis[1])
            affinity_list += [affinity]
            affinity_score = affinity_list[0]

        return affinity_score

    except Exception as e:
        print(f"Docking Failed: {smiles}")
        return 100.0



def reverse_sigmoid_transformation(original_score): 
    if original_score > 99.9:
        return -1.0 
    else: # return (0, 1)
        if original_score < -15:
            original_score = -12.5
        if original_score > -10:
            transformed = 1 / (1 + 10 ** (0.5 * (original_score + 10)))
        else:
            transformed = 1 / (1 + 10 ** (0.5 * (original_score + 10)))
        return transformed

def reverse_linear_transformation(original_score): 
    if original_score > 99.9:
        return -1.0 
    else: # return (0, 1)
        transformed = - original_score / 20
        return transformed

def generate_unique_label():
    timestamp = int(time.time() * 1000)
    unique_id = uuid.uuid4().hex
    return f"{timestamp}_{unique_id}"