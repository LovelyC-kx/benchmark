"""Load the existing model and generator without changing their weights."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'parent_l34'))
sys.path.insert(0,str(ROOT/'parent_l34/parent_l331/parent_l32/parent_l2'))
from bridge import load_model,flatten_contents
from parent_sampling import fixed_batch,repeat_panel
from factor_sampling import factor_schema
from task_moments import teachers,labels
from snnbench.neurons import Cell,NEURONS
from snnbench.models import BenchmarkModel,ModelConfig,BACKBONES
from snnbench.l2_runtime import StreamSession,clone_tree

