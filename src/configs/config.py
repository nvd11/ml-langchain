import yaml
import os
import sys
from loguru import logger

# append project path to sys.path
script_path = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))

print("project_path is {}".format(project_path))

# append project path to sys.path
sys.path.append(project_path)


# setup logs path
logger.add(os.path.join(project_path, "logs", "app.log"))

logger.info("basic setup done")


yaml_configs = None
# load additon configs.yaml
with open(os.path.join(project_path, "src", "configs", "config_dev.yaml")) as f:
    yaml_configs = yaml.load(f, Loader=yaml.FullLoader)

if "gemini" in yaml_configs and "api-key" in yaml_configs["gemini"]:
    api_key = os.environ.get(yaml_configs["gemini"]["api-key"])
    if api_key:
        yaml_configs["gemini"]["api-key"] = api_key 
        logger.info(f"EnvironmenT variable {yaml_configs['gemini']['api-key'][:5]}xxxxxx.... found, using the value from environment variable")
    else:
        logger.warning(f"Environment variable {yaml_configs['gemini']['api-key']} not found , using valud from config file")