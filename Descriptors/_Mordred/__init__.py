import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from mordred import Calculator, descriptors

from collections import defaultdict


HALOGENS = {'F','Cl','Br','I'}

# metals observed in your dataset (adjustable)
METAL_ELEMS = sorted(set([
    "As","Hg","Cd","Pb","Sn","Sb","Bi",
    "Cu","Zn","Ag","Au","Ni","Co","Cr","Mn","Fe",
    "Pt","Pd",
    "Be","Tl","V","Mo","W","Se","Te",
    "Zr","Eu","Dy"
]))

# ✅ 核心：Zn/Sn/Hg 同等对待，都“不强制进训练集”
NOT_FORCED_METALS = {"Zn", "Sn", "Hg"}

# ✅ 只有 RAW 中出现这些金属才“强制进训练集”
FORCE_METALS = [m for m in METAL_ELEMS if m not in NOT_FORCED_METALS]

TOXIC_METALS_FOCUS = ["Hg","Sn","Zn","Pb","Cd","As","Cr","Ni","Cu","Ag","Au","Mn","Fe","Bi","Zr","Eu","Dy"]


## Calculators
MORD_CALC = Calculator(descriptors, ignore_3D=True)
RD_NAMES = [d[0] for d in Descriptors._descList]
RD_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(RD_NAMES)


def mol_from_smiles_robust(smi: str):
    if smi is None or (isinstance(smi, float) and np.isnan(smi)):
        return None
    smi = str(smi).strip()
    if not smi:
        return None
    mol = None
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        mol = None

    return mol

def has_carbon(mol: Chem.Mol) -> bool:
    return mol is not None and any(a.GetSymbol() == "C" for a in mol.GetAtoms())

def metal_labels_from_mol(mol: Chem.Mol, prefix=""):

    labels = {f"{prefix}has_{m}": 0 for m in METAL_ELEMS}
    if mol is None:
        labels[f"{prefix}has_any_metal"] = 0
        labels[f"{prefix}n_metals"] = 0
        return labels

    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    present = []
    for m in METAL_ELEMS:
        if m in syms:
            labels[f"{prefix}has_{m}"] = 1
            present.append(m)
    labels[f"{prefix}has_any_metal"] = 1 if present else 0
    labels[f"{prefix}n_metals"] = len(present)
    return labels

def metal_set_from_labels(lab: dict, prefix=""):
    s = set()
    for m in METAL_ELEMS:
        if lab.get(f"{prefix}has_{m}", 0) == 1:
            s.add(m)
    return s


# Standardize SMILES: remove counter-ions (largest carbon fragment), keep charge, drop no-carbon

def largest_carbon_fragment(mol: Chem.Mol):
    """Pick the largest fragment that contains carbon. If none has carbon -> None."""
    if mol is None:
        return None
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except:
        frags_raw = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        frags = []
        for f in frags_raw:
            try:
                Chem.SanitizeMol(f)
                frags.append(f)
            except:
                pass

    carbon_frags = [f for f in frags if has_carbon(f)]
    if not carbon_frags:
        return None
    carbon_frags.sort(key=lambda m: (m.GetNumHeavyAtoms(), m.GetNumAtoms()), reverse=True)
    return carbon_frags[0]

def standardize_smiles_keep_charge_remove_counterions(raw_smi: str):
    """
    - Parse raw -> mol
    - Keep largest carbon-containing fragment
    - Keep charge (no uncharge)
    - If no carbon -> None
    """
    mol = mol_from_smiles_robust(raw_smi)
    if mol is None:
        return None
    core = largest_carbon_fragment(mol)
    if core is None:
        return None
    try:
        return Chem.MolToSmiles(core, canonical=True)
    except:
        return None

# Anchor features: raw & core metal tags + derived + physchem(core)
# =========================================================
def anchor_features(raw_smi: str, core_smi: str):
    feats = defaultdict(float)

    mol_raw = mol_from_smiles_robust(raw_smi)
    mol_core = mol_from_smiles_robust(core_smi)

    raw_lab = metal_labels_from_mol(mol_raw, prefix="raw_")
    core_lab = metal_labels_from_mol(mol_core, prefix="core_")

    feats.update(raw_lab)
    feats.update(core_lab)

    raw_set = metal_set_from_labels(raw_lab, prefix="raw_")
    core_set = metal_set_from_labels(core_lab, prefix="core_")

    feats["metal_only_raw"] = int(len(raw_set) > 0 and len(core_set) == 0)
    feats["raw_has_force_metal"] = int(any(m in raw_set for m in FORCE_METALS))

    # core physchem
    feats["has_charge_core"] = 0
    feats["has_aromatic_core"] = 0
    feats["n_heavy_atoms_core"] = 0
    feats["n_halogens_core"] = 0
    feats["mw_core"] = 0
    feats["logp_core"] = 0
    feats["tpsa_core"] = 0
    feats["hbd_core"] = 0
    feats["hba_core"] = 0

    if mol_core is not None:
        try:
            atoms = mol_core.GetAtoms()
            syms = [a.GetSymbol() for a in atoms]
            feats["has_charge_core"] = int(Chem.GetFormalCharge(mol_core) != 0)
            feats["has_aromatic_core"] = int(any(a.GetIsAromatic() for a in atoms))
            feats["n_heavy_atoms_core"] = mol_core.GetNumHeavyAtoms()
            feats["n_halogens_core"] = sum(s in HALOGENS for s in syms)

            feats["mw_core"] = Descriptors.MolWt(mol_core)
            feats["logp_core"] = Descriptors.MolLogP(mol_core)
            feats["tpsa_core"] = Descriptors.TPSA(mol_core)
            feats["hbd_core"] = Descriptors.NumHDonors(mol_core)
            feats["hba_core"] = Descriptors.NumHAcceptors(mol_core)
        except:
            pass

    return dict(feats)

def rdkit_features_core(core_smi: str):

    mol = mol_from_smiles_robust(core_smi)
    if mol is None:
        return {}
    try:
        vals = RD_CALC.CalcDescriptors(mol)
        return {f"rd_{k}": float(v) for k, v in zip(RD_NAMES, vals) if np.isfinite(v)}
    except:
        return {}

def mordred_features_full_core(core_smi: str):
    mol = mol_from_smiles_robust(core_smi)
    if mol is None:
        return {}
    try:
        desc = MORD_CALC(mol)
        feats = {}
        for k, v in desc.items():
            # if isinstance(v, (int, float)) and np.isfinite(v):
                feats[f"mord_{k}"] = float(v)
        return feats
    except:
        return {}
