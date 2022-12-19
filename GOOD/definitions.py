r"""
Only for project usage. This module includes a preset project root and a storage root.
"""
import os

# root outside the GOOD
STORAGE_DIR = "/apdcephfs/share_1364275/yandrewchen/GOOD/storage_cigab32_orig_modelfix" #os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'storage')  #: :str - Storage root=<Project root>/storage/.
ROOT_DIR = "/apdcephfs/private_yandrewchen/GOOD" #os.path.abspath(os.path.dirname(os.path.dirname(__file__)))  #: str - Project root or Repo root.
OOM_CODE = 88
