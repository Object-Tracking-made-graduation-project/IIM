from dataclasses import dataclass

model_path = r'./FDST-HR-ep_177_F1_0.969_Pre_0.984_Rec_0.955_mae_1.0_mse_1.5.pth'


@dataclass()
class ModelParams:
    weight_path: str = model_path
    gpu_id: str = '0'
    netName: str = 'HR_Net'


@dataclass()
class ImageParams:
    path: str = None
    output_dir: str = None


@dataclass()
class InferenceParams:
    model_params: ModelParams
    image_params: ImageParams = None
