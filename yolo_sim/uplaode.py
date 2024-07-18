from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = 'ce59102c-82eb-48f5-a9aa-ead7f6b78bc0'

api = HubApi()
api.login(YOUR_ACCESS_TOKEN)
api.push_model(
    model_id="LiuKai/yolo-sim",
    model_dir="D:\program\yolo_sim" # 本地模型目录，要求目录中必须包含configuration.json
)