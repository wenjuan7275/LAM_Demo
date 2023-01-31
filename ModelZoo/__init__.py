import os
import torch

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


NN_LIST = [
    'RCAN',
    'CARN',
    'RRDBNet',
    'RNAN',
    'SAN',
    'GRL'
]


MODEL_LIST = {
    'RCAN': {
        'Base': 'RCAN.pt',
    },
    'CARN': {
        'Base': 'CARN_7400.pth',
    },
    'RRDBNet': {
        'Base': 'RRDBNet_PSNR_SRx4_DF2K_official-150ff491.pth',
    },
    'SAN': {
        'Base': 'SAN_BI4X.pt',
    },
    'RNAN': {
        'Base': 'RNAN_SR_F64G10P48BIX4.pt',
    },
    'GRL':  {
        'Tiny': 'gnll_final_tiny_w32_s64.ckpt'
    }
}

def print_network(model, model_name):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f K. '
          'To see the architecture, do print(network).'
          % (model_name, num_params / 1000))


def get_model(model_name, factor=4, num_channels=3):
    """
    All the models are defaulted to be X4 models, the Channels is defaulted to be RGB 3 channels.
    :param model_name:
    :param factor:
    :param num_channels:
    :return:
    """
    print(f'Getting SR Network {model_name}')
    if model_name.split('-')[0] in NN_LIST:

        if model_name == 'RCAN':
            from .NN.rcan import RCAN
            net = RCAN(factor=factor, num_channels=num_channels)

        elif model_name == 'CARN':
            from .CARN.carn import CARNet
            net = CARNet(factor=factor, num_channels=num_channels)

        elif model_name == 'RRDBNet':
            from .NN.rrdbnet import RRDBNet
            net = RRDBNet(num_in_ch=num_channels, num_out_ch=num_channels)

        elif model_name == 'SAN':
            from .NN.san import SAN
            net = SAN(factor=factor, num_channels=num_channels)

        elif model_name == 'RNAN':
            from .NN.rnan import RNAN
            net = RNAN(factor=factor, num_channels=num_channels)
        elif model_name == 'GRL':
            from .NN.grl import GNLL
            net = GNLL(
                upscale=4,
                img_size=64,
                window_size=32,
                stripe_size=[64, 64],
                depths=[4, 4, 4, 4],
                embed_dim=64,
                num_heads_window=[2, 2, 2, 2],
                num_heads_stripe=[2, 2, 2, 2],
                mlp_ratio=2,
                qkv_proj_type="linear",
                anchor_proj_type="avgpool",
                anchor_window_down_factor=2,
                out_proj_type="linear",
                conv_type="1conv",
                upsampler="pixelshuffledirect",
                fairscale_checkpoint=True,
            )

        else:
            raise NotImplementedError()

        print_network(net, model_name)
        return net
    else:
        raise NotImplementedError()


def load_model(model_loading_name):
    """
    :param model_loading_name: model_name-training_name
    :return:
    """
    splitting = model_loading_name.split('@')
    if len(splitting) == 1:
        model_name = splitting[0]
        training_name = 'Base'
    elif len(splitting) == 2:
        model_name = splitting[0]
        training_name = splitting[1]
    else:
        raise NotImplementedError()
    assert model_name in NN_LIST or model_name in MODEL_LIST.keys(), 'check your model name before @'
    net = get_model(model_name)
    state_dict_path = os.path.join(MODEL_DIR, MODEL_LIST[model_name][training_name])
    print(f'Loading model {state_dict_path} for {model_name} network.')
    state_dict = torch.load(state_dict_path, map_location='cpu')
    if model_name == "GRL":
        new_state_dict = {}
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            for k, v in state_dict.items():
                if (
                        k.find("relative_coords_table") >= 0
                        or k.find("relative_position_index") >= 0
                        or k.find("attn_mask") >= 0
                        or k.find("model.table_") >= 0
                        or k.find("model.index_") >= 0
                        or k.find("model.mask_") >= 0
                        # or k.find(".upsample.") >= 0
                ):
                    print(k)
                elif k.find("model.") >= 0:
                    new_state_dict[k.replace("model.", "")] = v

        # try:
        current_state_dict = net.state_dict()
        current_state_dict.update(new_state_dict)
        net.load_state_dict(current_state_dict, strict=True)
    else:
        net.load_state_dict(state_dict)
    return net