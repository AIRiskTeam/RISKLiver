import pandas as pd
import numpy as np
from typing import Literal

# =========================================================
# helpers
# =========================================================
def safe_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, str) and x.strip() == "":
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def safe_int(x):
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        return int(float(x))
    except Exception:
        return None

def f_acid_neutral(pKa_a, pH=7.4):
    # HA <-> A- + H+ ; neutral is HA
    return 1.0 / (1.0 + 10.0 ** (pH - pKa_a))

def f_base_neutral(pKa_b, pH=7.4):
    # BH+ <-> B + H+ ; neutral is B
    return 1.0 / (1.0 + 10.0 ** (pKa_b - pH))

def choose_single_pka_mode(pKa_a, pKa_b, pH=7.4):
    """
    Armitage-style simplification for both pKa_a and pKa_b:
    choose the pKa closer to pH to decide acid vs base mode.
    Returns: (mode, chosen_pKa) where mode in {"neutral","acid","base"}.
    """
    pa = safe_float(pKa_a)
    pb = safe_float(pKa_b)

    if pd.isna(pa) and pd.isna(pb):
        return "neutral", np.nan
    if pd.notna(pa) and pd.isna(pb):
        return "acid", pa
    if pd.notna(pb) and pd.isna(pa):
        return "base", pb

    # both present
    if abs(pH - pa) <= abs(pH - pb):
        return "acid", pa
    else:
        return "base", pb

# ---- Neutral K from logKow (Toxics 2021)
def KSaW_neutral_from_logKow(logKow):
    if logKow < 4.5:
        return 10.0 ** (1.08 * logKow - 0.70)
    else:
        return 10.0 ** (0.37 * logKow + 2.56)

def KMW_neutral_from_logKow(logKow):
    return 10.0 ** (1.01 * logKow + 0.12)

def KSlW_neutral_from_logKow(logKow):
    return 10.0 ** (logKow)

def D_from_K_neutral_ion(KN, KI, f_neutral, f_ion_total):
    return f_neutral * KN + f_ion_total * KI

# =========================================================
# Calculator class
# =========================================================

class IVDCalculator():
    def __init__(self, endpoint:Literal['viability', 'mitotox', 'apoptosis']):
        # ---- columns
        self.LOGP_COL = "LogP_pred"          # use as logKow,N proxy (neutral)
        self.IC_COLS  = ["IC20"]

        self.ION_COL   = "ionization"        # 0/1/2 (your encoding)
        self.PKA_A_COL = "pKa_a_pred"        # acid pKa (may be empty)
        self.PKA_B_COL = "pKa_b_pred"        # base pKa (may be empty)

        self.PH = 7.4

        # If you don't want mass-balance QC columns, set this False
        self.DO_MASS_BALANCE_QC = True

        # =========================================================
        # 1) ASSAY SETTINGS
        # =========================================================
        ASSAY = {
            "viability": {"V_MEDIUM_uL":6.0,"N_CELLS":600,"FBS":0.10},
            "mitotox": {"V_MEDIUM_uL":5.0,"N_CELLS":2000,"FBS":0.10},
            "apoptosis": {"V_MEDIUM_uL":5.0,"N_CELLS":2000,"FBS":0.10}
            }
        self.ASSAY = ASSAY[endpoint]

        # =========================================================
        # 2) SYSTEM COMPOSITION (10% FBS medium + HepG2 cell fractions)
        # =========================================================
        self.VF_W_MED    = 0.9945
        self.VF_PROT_MED = 0.0052
        self.VF_LIP_MED  = 0.0003   # serum lipid fraction (neutral lipid phase)

        self.VF_W_CELL     = 0.8735
        self.VF_PROT_CELL  = 0.0946
        self.VF_MEM_CELL   = 0.02552
        self.VF_STOR_CELL  = 0.00638

        # per-cell volume (UPDATED per your data)
        self.V_CELL_PER_CELL_L = 3.0395e-15  # L/cell

        # =========================================================
        # 3) Ionic scaling factors (d-values; do NOT name them D_*)
        # =========================================================
        self.d_MW = 1.0        # for KMW ionic scaling
        self.d_OW = 3.1        # for octanol/neutral lipid (used for SlW ionic scaling)
        self.d_SAW_ACID = 0.0  # albumin: acids
        self.d_SAW_BASE = 1.3  # albumin: bases

        self.VOL = self.compute_assay_volumes()

    def f_neutral_and_d_saw(self, pKa_a, pKa_b, ion_code, pH=7.4):
        """
        Returns:
            f_neu, f_ion, d_saw, mode_used
        """
        ion_code = safe_int(ion_code)
        mode, pka = choose_single_pka_mode(pKa_a, pKa_b, pH=pH)

        if ion_code == 0 or mode == "neutral":
            return 1.0, 0.0, 0.0, "neutral"

        if mode == "acid":
            fN = float(np.clip(f_acid_neutral(pka, pH), 0.0, 1.0))
            return fN, 1.0 - fN, self.d_SAW_ACID, "acid"

        # mode == "base"
        fN = float(np.clip(f_base_neutral(pka, pH), 0.0, 1.0))
        return fN, 1.0 - fN, self.d_SAW_BASE, "base"

    def compute_assay_volumes(self):
        Vmed_L  = self.ASSAY["V_MEDIUM_uL"] * 1e-6
        Vcell_L = self.ASSAY["N_CELLS"] * self.V_CELL_PER_CELL_L
        Vsys_L  = Vmed_L + Vcell_L

        # medium sub-volumes
        Vw_med    = Vmed_L * self.VF_W_MED
        Vp_med    = Vmed_L * self.VF_PROT_MED
        Vl_med    = Vmed_L * self.VF_LIP_MED

        # cell sub-volumes
        Vw_cell   = Vcell_L * self.VF_W_CELL
        Vp_cell   = Vcell_L * self.VF_PROT_CELL
        Vmem_cell = Vcell_L * self.VF_MEM_CELL
        Vstor_cell= Vcell_L * self.VF_STOR_CELL

        return {
            "Vmed_L": Vmed_L, "Vcell_L": Vcell_L, "Vsys_L": Vsys_L,
            "Vw_med": Vw_med, "Vp_med": Vp_med, "Vl_med": Vl_med,
            "Vw_cell": Vw_cell, "Vp_cell": Vp_cell,
            "Vmem_cell": Vmem_cell, "Vstor_cell": Vstor_cell
        }

    def mass_balance_qc(self, Cnom, Cfree, D_SaW, D_SlW, D_MW) -> tuple[float, float]:
        if (pd.isna(Cnom) or pd.isna(Cfree) or
            pd.isna(D_SaW) or pd.isna(D_SlW) or pd.isna(D_MW)):
            return np.nan, np.nan

        # medium
        M_w_med = Cfree * self.VOL["Vw_med"]
        M_p_med = Cfree * D_SaW * self.VOL["Vp_med"]
        M_l_med = Cfree * D_SlW * self.VOL["Vl_med"]

        # cell
        M_w_cell = Cfree * self.VOL["Vw_cell"]
        M_p_cell = Cfree * D_SaW * self.VOL["Vp_cell"]
        M_mem    = Cfree * D_MW  * self.VOL["Vmem_cell"]
        M_stor   = Cfree * D_SlW * self.VOL["Vstor_cell"]

        M_sum = M_w_med + M_p_med + M_l_med + M_w_cell + M_p_cell + M_mem + M_stor
        M_tot = Cnom * self.VOL["Vsys_L"]

        if M_tot == 0:
            return np.nan, np.nan
        return M_sum / M_tot, M_sum - M_tot

    def compute_row(self, row):
        logKow = safe_float(row.get(self.LOGP_COL, np.nan))
        ion_code = safe_int(row.get(self.ION_COL, None))
        pKa_a = row.get(self.PKA_A_COL, np.nan)
        pKa_b = row.get(self.PKA_B_COL, np.nan)

        out = {}

        if pd.isna(logKow):
            # fill NA outputs
            base_cols = [
                "logKow_N","mode_used","f_neutral_74","f_ion_74",
                "D_SaW","D_SlW","D_MW","Dmed_w","Dcell_w","Dmed_cell",
                "f_cell","f_med","f_free_medium","f_mem",
                "Vsystem_L","Vw_med_L","Vcell_L","Vmem_cell_L"
            ]
            for k in base_cols:
                out[k] = np.nan

            for ic in self.IC_COLS:
                out[f"{ic}_free"] = np.nan
                out[f"{ic}_cell"] = np.nan
                out[f"{ic}_mem"]  = np.nan
                if self.DO_MASS_BALANCE_QC:
                    out[f"MB_ratio_{ic}"] = np.nan
                    out[f"MB_err_{ic}"]   = np.nan

            for suf in ["free","cell","mem"]:
                out[f"logIC20_{suf}"] = np.nan

            return pd.Series(out)

        # -------- speciation + albumin scaling choice
        f_neu, f_ion, d_saw, mode_used = self.f_neutral_and_d_saw(pKa_a, pKa_b, ion_code, pH=self.PH)

        # -------- neutral K
        KSaW_N = KSaW_neutral_from_logKow(logKow)
        KMW_N  = KMW_neutral_from_logKow(logKow)
        KSlW_N = KSlW_neutral_from_logKow(logKow)

        # -------- ionic K (use d_* constants; no name collision)
        KSaW_I = KSaW_N * (10.0 ** (-d_saw))
        KMW_I  = KMW_N  * (10.0 ** (-self.d_MW))
        KSlW_I = KSlW_N * (10.0 ** (-self.d_OW))

        # -------- D(pH)
        D_SaW = D_from_K_neutral_ion(KSaW_N, KSaW_I, f_neu, f_ion)
        D_MW  = D_from_K_neutral_ion(KMW_N,  KMW_I,  f_neu, f_ion)
        D_SlW = D_from_K_neutral_ion(KSlW_N, KSlW_I, f_neu, f_ion)

        # -------- Fabian eq(3)/(4) style distribution ratios
        Dmed_w = self.VF_W_MED + self.VF_PROT_MED * D_SaW + self.VF_LIP_MED * D_SlW
        Dcell_w = self.VF_W_CELL + self.VF_PROT_CELL * D_SaW + self.VF_MEM_CELL * D_MW + self.VF_STOR_CELL * D_SlW

        # -------- cell vs medium (Fabian eq7-9)
        Dmed_cell = (Dmed_w / Dcell_w) if (pd.notna(Dmed_w) and pd.notna(Dcell_w) and Dcell_w > 0) else np.nan
        if pd.notna(Dmed_cell) and self.VOL["Vcell_L"] > 0:
            f_cell = 1.0 / (1.0 + Dmed_cell * (self.VOL["Vmed_L"] / self.VOL["Vcell_L"]))
            f_med  = 1.0 - f_cell
        else:
            f_cell, f_med = np.nan, np.nan

        # -------- f_free_medium (Fabian sink term)
        Vw_med = self.VOL["Vw_med"]
        f_free = 1.0 / (
            1.0
            + D_SaW * (self.VOL["Vp_med"] / Vw_med)
            + D_SlW * (self.VOL["Vl_med"] / Vw_med)
            + Dcell_w * (self.VOL["Vcell_L"] / Vw_med)
        )

        # -------- f_mem (Fabian eq11 structure; membrane-only lipid = membrane lipid)
        Vmem = self.VOL["Vmem_cell"]
        if Vmem > 0 and D_MW > 0:
            f_mem = 1.0 / (
                1.0
                + (1.0 / D_MW) * (self.VOL["Vw_cell"] / Vmem)
                + (D_SaW / D_MW) * (self.VOL["Vp_cell"] / Vmem)
                + (Dmed_w / D_MW) * (self.VOL["Vmed_L"] / Vmem)
            )
        else:
            f_mem = np.nan

        # -------- outputs (meta)
        out.update({
            "logKow_N": logKow,
            "mode_used": mode_used,
            "f_neutral_74": f_neu,
            "f_ion_74": f_ion,
            "D_SaW": D_SaW,
            "D_SlW": D_SlW,
            "D_MW": D_MW,
            "Dmed_w": Dmed_w,
            "Dcell_w": Dcell_w,
            "Dmed_cell": Dmed_cell,
            "f_cell": f_cell,
            "f_med": f_med,
            "f_free_medium": f_free,
            "f_mem": f_mem,
            "Vsystem_L": self.VOL["Vsys_L"],
            "Vw_med_L": self.VOL["Vw_med"],
            "Vcell_L": self.VOL["Vcell_L"],
            "Vmem_cell_L": self.VOL["Vmem_cell"],
        })

        # -------- convert ICs + optional mass balance QC
        for ic in self.IC_COLS:
            Cnom = safe_float(row.get(ic, np.nan))
            if pd.isna(Cnom):
                out[f"{ic}_free"] = np.nan
                out[f"{ic}_cell"] = np.nan
                out[f"{ic}_mem"]  = np.nan
                if self.DO_MASS_BALANCE_QC:
                    out[f"MB_ratio_{ic}"] = np.nan
                    out[f"MB_err_{ic}"]   = np.nan
                continue

            # Cfree (M) in medium water
            Cfree = Cnom * f_free * (self.VOL["Vsys_L"] / self.VOL["Vw_med"])
            out[f"{ic}_free"] = Cfree

            # Ccell (M) in cell (relative to cell water)
            out[f"{ic}_cell"] = Cfree * Dcell_w if pd.notna(Cfree) else np.nan

            # Cmem (M) in membrane lipid phase
            if pd.notna(f_mem) and f_mem > 0 and self.VOL["Vmem_cell"] > 0:
                out[f"{ic}_mem"] = Cnom * f_mem * (self.VOL["Vsys_L"] / self.VOL["Vmem_cell"])
            else:
                out[f"{ic}_mem"] = np.nan

            if self.DO_MASS_BALANCE_QC:
                ratio, err = self.mass_balance_qc(Cnom, Cfree, D_SaW, D_SlW, D_MW)
                out[f"MB_ratio_{ic}"] = ratio
                out[f"MB_err_{ic}"]   = err

        # logs
        for suffix in ["free", "cell", "mem"]:
            v = out.get(f"IC20_{suffix}", np.nan)
            out[f"logIC20_{suffix}"] = np.log10(v) if (pd.notna(v) and v > 0) else np.nan

        return pd.Series(out)


def calculate_ivd(opera_result:pd.DataFrame, endpoint:str, ic20:float):
    df = opera_result
    df['IC20'] = ic20
    ivd_calculator = IVDCalculator(endpoint=endpoint)
    corr = df.apply(ivd_calculator.compute_row, axis=1)

    for c in corr.columns:
        if c in df.columns:
            df[c + "_new"] = corr[c]
        else:
            df[c] = corr[c]
    return df
