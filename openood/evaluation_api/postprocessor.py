import os
import urllib.request

from openood.postprocessors import *
from openood.utils.config import Config, merge_configs

postprocessors = {
    'ash': ASHPostprocessor,
    'ash_s': ASHPostprocessor,
    'cider': CIDERPostprocessor,
    'conf_branch': ConfBranchPostprocessor,
    'msp': BasePostprocessor,
    'ebo': EBOPostprocessor,
    'ebo_own': EBOPostprocessor,
    'odin': ODINPostprocessor,
    'odin_mod': ODINPostprocessor,
    'odin_nopert': ODINPostprocessor,
    'odin_notemp': ODINPostprocessor,
    'mds': MDSPostprocessor,
    'mds_ensemble': MDSEnsemblePostprocessor,
    'mds_ensemble_mod2': MDSEnsemblePostprocessorMod,
    'npos': NPOSPostprocessor,
    'rmds': RMDSPostprocessor,
    'gmm': GMMPostprocessor,
    'patchcore': PatchcorePostprocessor,
    'openmax': OpenMax,
    'react': ReactPostprocessor,
    'react_mod': ReactPostprocessor,
    'vim': VIMPostprocessor,
    'gradnorm': GradNormPostprocessor,
    'godin': GodinPostprocessor,
    'mds': MDSPostprocessor,
    'gram': GRAMPostprocessor,
    'gram_mod': GRAMPostprocessorMod,
    'cutpaste': CutPastePostprocessor,
    'mls': MaxLogitPostprocessor,
    'residual': ResidualPostprocessor,
    'klm': KLMatchingPostprocessor,
    'temp_scaling': TemperatureScalingPostprocessor,
    'ensemble': EnsemblePostprocessor,
    'dropout': DropoutPostProcessor,
    'draem': DRAEMPostprocessor,
    'dsvdd': DSVDDPostprocessor,
    'mos': MOSPostprocessor,
    'mcd': MCDPostprocessor,
    'opengan': OpenGanPostprocessor,
    'knn': KNNPostprocessor,
    'dice': DICEPostprocessor,
    'dice_mod': DICEPostprocessor,
    'ssd': SSDPostprocessor,
    'she': SHEPostprocessor,
    'rd4ad': Rd4adPostprocessor,
    'rotpred': RotPredPostprocessor,
    'rankfeat': RankFeatPostprocessor,
    'gen': GENPostprocessor
}

link_prefix = 'https://raw.githubusercontent.com/Jingkang50/OpenOOD/main/configs/postprocessors/'


def get_postprocessor(config_root: str, postprocessor_name: str,
                      id_data_name: str):
    postprocessor_config_path = os.path.join(config_root, 'postprocessors',
                                             f'{postprocessor_name}.yml')
    if not os.path.exists(postprocessor_config_path):
        os.makedirs(os.path.dirname(postprocessor_config_path), exist_ok=True)
        urllib.request.urlretrieve(link_prefix + f'{postprocessor_name}.yml',
                                   postprocessor_config_path)

    config = Config(postprocessor_config_path)
    config = merge_configs(config,
                           Config(**{'dataset': {
                               'name': id_data_name
                           }}))
    postprocessor = postprocessors[postprocessor_name](config)
    postprocessor.APS_mode = config.postprocessor.APS_mode
    postprocessor.hyperparam_search_done = False
    return postprocessor
