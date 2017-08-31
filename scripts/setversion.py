import sys

def replace_line(line, marker, newstr):
    if line.lstrip().startswith(marker):
        return newstr
    else:
        return line

def replace_in_file(fname, marker, newstr):
    with open(fname) as f:
        lines = list(f)
    lines = [replace_line(l, marker, newstr) for l in lines]
    with open(fname, 'w') as f:
        f.write(''.join(lines))

if __name__ == '__main__':
    version = sys.argv[1]
    replace_in_file('setup.py', 'version =', "    version = '%s',\n" % version)
    replace_in_file('pyiga/__init__.py', '__version__ =', "__version__ = '%s'\n" % version)
    replace_in_file('docs/source/conf.py', 'version =', "version = '%s'\n" % version)
