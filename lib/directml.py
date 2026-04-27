"""DirectML setup and workarounds.

Import this module BEFORE importing ultralytics. It applies two patches:

1. Aliases ``torch.inference_mode`` to ``torch.no_grad``. Ultralytics wraps
   ``predict()`` in ``@smart_inference_mode``, but DirectML can't set
   ``version_counter`` on inference-mode tensors.

2. Wraps ``ultralytics.utils.nms.non_max_suppression`` to move predictions to
   CPU just for NMS. DirectML fails on ``torch.cat`` + boolean indexing inside
   NMS; inference itself still runs on the GPU.

Use ``get_device(device_type)`` to resolve a device handle.
"""
import torch

torch.inference_mode = torch.no_grad

import ultralytics.utils.nms as _nms_mod

_orig_nms = _nms_mod.non_max_suppression


def _cpu_nms(prediction, *args, **kwargs):
    if isinstance(prediction, (list, tuple)):
        prediction = type(prediction)(
            p.detach().cpu() if torch.is_tensor(p) else p for p in prediction
        )
    elif torch.is_tensor(prediction):
        prediction = prediction.detach().cpu()
    return _orig_nms(prediction, *args, **kwargs)


_nms_mod.non_max_suppression = _cpu_nms


def get_device(device_type: str):
    """Resolve a device handle for the given device type.

    For DirectML, prefers a discrete GPU (RX/RTX/Arc/etc.) over an integrated
    one when multiple adapters are available.
    """
    if device_type == "directml":
        import torch_directml

        n = torch_directml.device_count()
        print(f"DirectML adapters found: {n}")
        for i in range(n):
            print(f"  [{i}] {torch_directml.device_name(i)}")

        chosen = 0
        discrete_keywords = ("rx", "radeon pro", "arc", "geforce", "rtx")
        for i in range(n):
            name = torch_directml.device_name(i).lower()
            if any(k in name for k in discrete_keywords):
                chosen = i
                break

        device = torch_directml.device(chosen)
        print(f"Selected DirectML adapter [{chosen}]: {torch_directml.device_name(chosen)}")
        return device

    if device_type == "cuda":
        return "cuda"

    return "cpu"
