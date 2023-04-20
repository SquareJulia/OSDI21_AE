from reorder import ReorderStrategy
SPARSERT_DIR = '../SparseRT/'
SPARSERT_MATERIALS_DIR = '{}materials/'.format(SPARSERT_DIR)
SPARSERT_MATERIALS_DATA_DIR = '{}data/'.format(SPARSERT_MATERIALS_DIR)
SPARSERT_MATERIALS_DIST_DIR = '{}dist/'.format(SPARSERT_MATERIALS_DIR)
SPARSERT_MATERIALS_TEMP_DIR = '{}temp/'.format(SPARSERT_MATERIALS_DIR)

PREPROCESSED_DIR = '../preprocessed'
PREPROCESSED_DATASET = 'dataset.pt'
PREPROCESSED_INPUT_INFO = 'inputInfo.pt'
PREPROCESSED_INPUT_LAYER_SPRT = 'inputLayerSpRT.pt'
PREPROCESSED_HIDDEN_LAYER_SPRT = 'hiddenLayerSpRT.pt'

ReorderStrategyAlias = {
    ReorderStrategy['NONE']: 'no',
    ReorderStrategy['RANDOM']: 'rd',
    ReorderStrategy['DEGREE']: 'dg',
    ReorderStrategy['RABBIT']: 'rb',
    ReorderStrategy['METIS']: 'mt'
}


def pre_dir_data_template(path):
    path_seed = ''.join(path.split('.')[:-1])
    return PREPROCESSED_DIR+path_seed+'/'  # data


def pre_dir_params_template(inputInfo):
    return pre_dir_params_template_base(inputInfo.num_features, inputInfo.outputDim_input, inputInfo.outputDim_hidden,
                                        inputInfo.reorder_strategy, inputInfo.density, inputInfo.A_tileDim)


def pre_dir_params_template_base(num_features, hidden, num_classes, reorder_strategy, density, tileDim):
    strategy = ReorderStrategyAlias[reorder_strategy]
    return '{}_{}_{}_{}_{}_{}/'.format(num_features, hidden, num_classes, strategy, density, tileDim)
