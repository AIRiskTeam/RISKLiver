import tortoise
from tortoise import Tortoise
from . import get_config
from urllib.parse import quote
from .db_models import *

async def init_db():
    '初始化数据库'
    user = quote(get_config()['PG']['User'])
    passwd = quote(get_config()['PG']['Passwd'])
    db_host = quote(get_config()['PG']['Host'])
    db_port = get_config()['PG']['Port']
    database = quote(get_config()['PG']['Database'])
    config_db = {
            "connections": {
                "riskliver_db": f"postgres://{user}:{passwd}@{db_host}:{db_port}/{database}"
            },
            "apps": {
                "riskliver_app": {
                    "models": ["RISKLiverServer.db_models"],
                    "default_connection": "riskliver_db",
                }
            },
        }
    await Tortoise.init(config_db, _enable_global_fallback=True)
    await Tortoise.generate_schemas(safe=True)
    print("DB initialized.")

async def close():
    '关闭数据库连接'
    await tortoise.connections.close_all(discard=False)
