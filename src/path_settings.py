import os,sys

# append project path to sys.path
script_path = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(script_path))

print("project_path is {}".format(project_path))

# append project path to sys.path
sys.path.append(project_path)