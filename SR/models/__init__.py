from .resnet import *
from .lstm import *

# KWT is an optional backbone. Its module (kwt.py) is not included in this
# snapshot; import it lazily so the ResNet/LSTM paths remain usable.
try:
    from .kwt import *  # noqa: F401,F403
    _HAS_KWT = True
except ImportError:
    _HAS_KWT = False

available_models = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'lstm',
    'kwt',
]

def create_model(model_name, num_classes, in_channels):
    if model_name in {'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'}:
        model = globals()[model_name](num_classes=num_classes, in_channels=in_channels)
    elif model_name == 'lstm':
        model = lstm(num_classes=num_classes, in_channels=in_channels)
    elif model_name == 'kwt':
        if not _HAS_KWT:
            raise ImportError(
                "The 'kwt' backbone requires SR/models/kwt.py, which is not "
                "included in this release. Use 'resnet18' or 'lstm' instead."
            )
        model = KWT(input_res=[80,87],patch_res=[80,1], num_classes=num_classes, dim=64, depth=12, heads=1, mlp_dim=256, pre_norm=True,emb_dropout = 0.1)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Available: {', '.join(available_models)}")
    return model
