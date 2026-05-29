import sys
from utils import *

try:
    recalls_data = load_recall_data()
    precisions_data = load_precision_data()
except FileNotFoundError as e:
    print("Required data not found in data directory. Refer to README.md to generate data.")
    sys.exit()

plot_project_link_percent()
plot_recall_bars(recalls_data, 1)
plot_precision_bars(precisions_data, 1)
plot_average_recall_bars(recalls_data, 1)
plot_average_precision_bars(precisions_data, 1)