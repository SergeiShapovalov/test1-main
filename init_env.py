import sys
import os


print("Current working directory:", os.getcwd())
print("sys.path before:", sys.path)

sys.path.extend(["/src"])

print("sys.path after:", sys.path)
print("Checking if directory exists:", os.path.exists("/src"))
print("Listing directory:", os.listdir("/src") if os.path.exists("/src") else "Directory does not exist")

from modules import launch_utils

with launch_utils.startup_timer.subcategory("prepare environment"):
    launch_utils.prepare_environment()
