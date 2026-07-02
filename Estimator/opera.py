import os, json, asyncio, sys
import pandas as pd
import numpy as np

from Descriptors.neutralize_new import neutralize_single_smiles

async def opera_predict(smiles_array:np.ndarray, hash:str):
    '调用opera预测IC20'
    # 读取配置
    with open("config.json", "r") as f:
        config = json.load(f)
    tmp_dir = config.get("tmpDir", "/dev/shm/riskliver")
    opera_executable = config.get("OPERAExecutable", "/usr/OPERA/application/run_OPERA_P.sh")
    matlab_runtime_dir = config.get("MATLABRuntimeDir", "/usr/local/MATLAB/MATLAB_Runtime/v912")
    
    # 创建临时文件
    os.makedirs(tmp_dir, exist_ok=True)
    TMP_INPUT = os.path.join(tmp_dir, f"{hash}.smi")
    TMP_OUTPUT = os.path.join(tmp_dir, f"{hash}.csv")

    # 写入smiles
    smi_list = [f"{neutralize_single_smiles(i)}\tSMI-{idx}\n" for idx, i in enumerate(smiles_array)] # writelines不会自动给行尾加\n
    with open(TMP_INPUT, "w", encoding='utf-8') as f:
        # smi = "\n".join(smi_list)
        # f.write(smi)
        f.writelines(smi_list)

    # 运行
    cmd = f"{opera_executable} {matlab_runtime_dir} -s {TMP_INPUT} -o {TMP_OUTPUT} -logP -logD -pKa -v 2 -c"
    p = await asyncio.subprocess.create_subprocess_shell(
        cmd, 
        stdout=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    p_in = "".encode(encoding='utf=8')
    p_out, p_err = await p.communicate(p_in)
    exit_code = p.returncode

    # 打印日志
    p_out = p_out.decode(encoding='utf-8')
    p_err = p_err.decode(encoding='utf-8')
    print(p_out, file=sys.stdout)
    print(p_err, file=sys.stderr)

    if exit_code != 0:
        raise RuntimeError(f"OPERA raises an error. \nInput: {smi_list}\nOutput: {p_out+p_err}")
    
    # 读取结果
    return_csv = pd.read_csv(TMP_OUTPUT)
    os.remove(TMP_INPUT)
    os.remove(TMP_OUTPUT)
    return return_csv

