from git import Repo
from os.path import dirname, abspath

_stage_mapping = {
    "origin/alpha": "alpha",
    "origin/beta": "beta",
    "origin/master": "prod",
    "origin/digital_square": "digi",
}

_sys_path_mapping = {
    "origin/alpha": "/var/www/PATTIE_User_Modeling_alpha",
    "origin/beta": "/var/www/PATTIE_User_Modeling_beta",
    "origin/master": "/var/www/PATTIE_User_Modeling",
    "origin/digital_square": "/var/www/PATTIE_Digital_Square",
}

path = dirname(dirname(abspath(__file__)))
print(path)
STAGE = ""
# _repo = Repo(path)

# _remote_branch_name = _repo.heads.digital_square.tracking_branch().name

# STAGE = _stage_mapping[_remote_branch_name]
# SYS_PATH = _sys_path_mapping[_remote_branch_name]
