import sys
import os
from glob import glob
import subprocess
import tempfile

import nbformat

def run_notebook(path, timeout=60):
    """Execute a notebook via nbconvert and collect output.
       :returns (parsed nb object, execution errors)
    """
    with tempfile.NamedTemporaryFile(suffix=".ipynb", mode='w+') as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
          "--ExecutePreprocessor.timeout=%d" % timeout,
          "--output", fout.name, path]
        subprocess.check_call(args)

        fout.seek(0)
        nb = nbformat.read(fout, nbformat.current_nbformat)

    errors = [output for cell in nb.cells if "outputs" in cell
                     for output in cell["outputs"]\
                     if output.output_type == "error"]

    return nb, errors

if __name__ == '__main__':
    # move to directory of this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    nbfiles = glob('notebooks/*ipynb')

    count = count_err = 0

    for nbfile in nbfiles:
        print('========== Running', nbfile, '==========')
        nb, errors = run_notebook('notebooks/subspace-correction-mg.ipynb', timeout=120)
        count += 1
        if errors:
            count_err += 1
            print('Errors', errors)
        else:
            print('No errors.')
    print('=====================================')
    print('Ran %d notebooks, %d had errors.' % (count, count_err))
    sys.exit(1 if count_err else 0)
