import sys
import runpy
import os

os.chdir(r'C:\Users\DELL\Desktop\MyProject\Kaggle\dong\\3')

args = "python -m torch.distributed.launch --nnodes=1 --nproc_per_node=1 main.py"

args = args.split()
if args[0] == 'python':
    """pop up the first in the args"""
    args.pop(0)

if args[0] == '-m':
    """pop up the first in the args"""
    args.pop(0)
    fun = runpy.run_module
else:
    fun = runpy.run_path

sys.argv.extend(args[1:])

fun(args[0], run_name='__main__')