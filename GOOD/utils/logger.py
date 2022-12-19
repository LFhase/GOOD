r"""A logger related utils file: tqdm style, logger loader.
"""
import os
from datetime import datetime
from cilog import create_logger
from torch.utils.tensorboard import SummaryWriter

pbar_setting = {'colour': '#a48fff', 'bar_format': '{l_bar}{bar:20}{r_bar}',
                'dynamic_ncols': True, 'ascii': '░▒█'}

from GOOD.utils.config_reader import Union, CommonArgs, Munch


def load_logger(config: Union[CommonArgs, Munch], sub_print=True):
    r"""
    Logger loader

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.log_path`, :obj:`config.tensorboard_logdir`, :obj:`config.log_file`)
        sub_print (bool): Whether the logger substitutes general print function. If Ture, logger.info will be equal to
            print(f'#IN#Message'), where #IN# represents info. Similarly, other level of log can be used by adding prefixes
            (Not capital sensitive): Debug: #d#, #De#, #Debug#, etc. Info: #I#, #In#, #inf#, #INFO#, etc. Important: #IM#,
            #important#, etc. Warning: #W#, #war#, etc. Error: #E#, #err#, etc. Critical: #C#, #Cri#, #critical#, etc. If
            there is no prefix, the general print function will be used.

    Returns:
        [cilog Logger, tensorboard summary writer]

    """
    if sub_print:
        print("This logger will substitute general print function")
    logger = create_logger(name='GNN_log',
                           file=config.log_path,
                           enable_mail=False,
                           sub_print=sub_print)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    writer = SummaryWriter(
        log_dir=os.path.join(config.tensorboard_logdir, f'{config.log_file}_{current_time}'))
    return logger, writer



def print_env():
    import sys
    import numpy as np
    import torch
    print("#IN#Environment:")
    if torch.cuda.device_count()>0:
        print("#IN#GPU: {}".format(torch.cuda.get_device_name(torch.cuda.current_device())))
    else:
        print("#IN#GPU: CPU : (")
    print("#IN#Python: {}".format(sys.version.split(" ")[0]))
    print("#IN#PyTorch: {}".format(torch.__version__))
    print("#IN#CUDA: {}".format(torch.version.cuda))
    print("#IN#CUDNN: {}".format(torch.backends.cudnn.version()))
    print("#IN#NumPy: {}".format(np.__version__))
