import os
import sys
import subprocess
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "stylegan-v"))

# from /third_party/stylegan-v/src/scripts/calc_metrics_for_dataset.py
# @click.option('--metrics', help='Comma-separated list or "none"', type=CommaSeparatedList(), default='fvd2048_16f,fid50k_full', show_default=True)
# @click.option('--real_data_path', help='Dataset to evaluate metrics against (directory or zip) [default: same as training data]', metavar='PATH')
# @click.option('--fake_data_path', help='Generated images (directory or zip)', metavar='PATH')
# @click.option('--mirror', help='Should we mirror the real data?', type=bool, metavar='BOOL')
# @click.option('--resolution', help='Resolution for the source dataset', type=int, metavar='INT')
# @click.option('--gpus', help='Number of GPUs to use', type=int, default=1, metavar='INT', show_default=True)
# @click.option('--verbose', help='Print optional information', type=bool, default=False, metavar='BOOL', show_default=True)
# @click.option('--use_cache', help='Use stats cache', type=bool, default=True, metavar='BOOL', show_default=True)
# @click.option('--num_runs', help='Number of runs', type=int, default=1, metavar='INT', show_default=True)
def call_script():
    args = sys.argv[1:]
    dirname = os.path.join(os.path.dirname(__file__), "../../third_party/stylegan-v")
    cmd = ["python", os.path.join(dirname, "src/scripts/calc_metrics_for_dataset.py")] + args
    subprocess.run(cmd, cwd=dirname)

if __name__ == "__main__":
    call_script()