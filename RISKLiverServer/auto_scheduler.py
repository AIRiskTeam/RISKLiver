import re, json, asyncio, traceback, sys, os, psutil
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from .db_models import Task
from . import get_config

from Estimator.streamlit import hazard_identification, regression_modeling, calculate_descriptors

async def get_device_info(device_types:list[str]) -> list[dict]:
    '获取GPU可用信息'
    device_list = []
    for device_type in device_types:
        if device_type in ("nvidia", "cuda"):
            # 叫一个nvidia-smi
            p = await asyncio.subprocess.create_subprocess_shell("nvidia-smi", stdout=asyncio.subprocess.PIPE)
            await p.wait()
            nvidia_output = (await p.stdout.read()).decode(encoding='utf-8')

            # 匹配字符串
            cuda_id_lines = re.finditer(
                pattern=r"(?<=\|)\s+\d+(\s|\d|\w|\.)+(?=(Off|On)\s+\|)", # NVIDIA GeForce RTX 5090 ...
                string=nvidia_output
            )
            usage_lines = re.findall(
                pattern=r"(?<=\|)\s+\d+\%\s+\S+(?=\s+\|\n)",
                string=nvidia_output
            )

            # 提取信息
            for idx, match in enumerate(cuda_id_lines):
                i = match.group().strip()
                d = dict(
                    name=f"cuda:{i.split(' ')[0]}",
                    model=" ".join(i.split(' ')[1:]).strip(),
                    usage=int(usage_lines[idx].split('%')[0]),
                )
                device_list.append(d)
        elif device_type == "cpu":
            # 提取CPU数量和load average
            cpu_count = os.cpu_count()
            a1, a2, a3 = psutil.getloadavg()

            # 提取load average
            d = dict(
                    name="cpu",
                    model="cpu",
                    usage=a1 / cpu_count * 100,
                )
            device_list.append(d)
        else:
            print(f"Warning: unsupported device type {device_type}.", file=sys.stderr)
    return sorted(device_list, key=lambda x:x['usage'])

async def get_most_usable_device() -> str:
    '获取最能用的设备'
    # 允许使用的GPU列表
    config = get_config()
    device_types = config.get("DeviceType", ["cpu"])

    # 查找可用的GPU
    device_list = await get_device_info(device_types)
    for device_info in device_list:
        # 可用时记录
        device_name = device_info['name']
        device_usage = device_info['usage']
        device_model = device_info['model']
        if device_usage <= 5:
            print(f"Allocating device {device_name} - {device_model}")
            return device_name
    else:
        return ""
    
async def _run_task(task, **kwargs):
    '运行任务'
    return task(**kwargs)

async def scheduled_check():
    '定时检查'
    FUNC_DICT = dict(
        calculate_descriptors=calculate_descriptors,
        hazard_identification=hazard_identification,
        regression_modeling=regression_modeling,
    )
    FUNC_KEYS = list(FUNC_DICT.keys())

    # 查询等待中的任务
    job_list = await Task.filter(status=0, task_name__in=FUNC_KEYS).values(
        'status', 'hash', 'task_name', 'kwargs'
        )
    # print(f"job_list: {job_list}")
    for _, task_dict in enumerate(job_list):
        # 分配GPU
        device = await get_most_usable_device()
        if device != "":
            # 获取到可用GPU时继续运行
            task_name = task_dict['task_name']
            kwargs = json.loads(task_dict['kwargs'])
            func = FUNC_DICT[task_name]
            hash = task_dict['hash']
            # 打印一下
            print(f"Received task {hash} ({task_name}).")
            await Task.filter(hash=hash).update(status=1)
            # 创建任务
            # task = asyncio.create_task(_run_task(func, device=device, **kwargs))
            # result = await task
            try:
                # result = func(device=device, **kwargs)
                result = await func(device=device, hash=hash, **kwargs)
                # when complete
                result_str = json.dumps(result.to_dict())
                await Task.filter(hash=hash).update(status=2, result=result_str)
            except Exception:
                # when error
                result = traceback.format_exc()
                print(result, file=sys.stderr)
                await Task.filter(hash=hash).update(status=3, result=result)

async def init():
    '初始化'
    INTERVAL = 10
    scheduler = AsyncIOScheduler()
    scheduler.add_job(scheduled_check, trigger="interval", seconds=INTERVAL)
    scheduler.start()
    print(f"Autobackup scheduler started (Interval: {INTERVAL}s)")
    return scheduler

