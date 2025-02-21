import platformdirs
import os
import shutil

if __name__ == '__main__':
    MODDIR = os.path.join(platformdirs.user_cache_dir('pyiga'), 'modules')
    print('Removing everything under', MODDIR)
    shutil.rmtree(MODDIR)
