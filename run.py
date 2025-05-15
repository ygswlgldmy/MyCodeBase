import sys
import runpy
import os

os.chdir('/Users/user/Desktop/MyCode')

args = "python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  main.py"

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