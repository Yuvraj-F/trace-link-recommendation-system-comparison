import sys
from utils import *

try:
    recalls_data = load_recall_data()
    precisions_data = load_precision_data()
except FileNotFoundError as e:
    print("Required data not found in data directory. Refer to README.md to generate data.")
    sys.exit()

plot_projects(recalls_data, precisions_data)
# plot_average_across_projects(recalls_data, precisions_data)