import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import tqdm
from model.locator import Crowd_locator
from misc.utils import *
from PIL import Image, ImageOps
import  cv2 
from collections import OrderedDict
from datetime import datetime

GPU_ID = '0'

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

torch.backends.cudnn.benchmark = True
netName = 'HR_Net'  # options: HR_Net,VGG16_FPN
model_path = './FDST-HR-ep_177_F1_0.969_Pre_0.984_Rec_0.955_mae_1.0_mse_1.5.pth'
image_file = "./figures/IMG_3.jpg"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


mean_std = ([0.452016860247, 0.447249650955, 0.431981861591], [0.23242045939, 0.224925786257, 0.221840232611])  

img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
restore = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])





def main():

    net = create_model(model_path)
    print("Model loaded")

    img = Image.open(image_file)

    start_time = datetime.now()

    jojo = predict_people_in_frame(img, net)

    print("inference time: ", datetime.now() - start_time)

    print("done:  ", jojo)









def get_boxInfo_from_Binar_map(Binar_numpy, min_area=3):
    Binar_numpy = Binar_numpy.squeeze().astype(np.uint8)
    assert Binar_numpy.ndim == 2
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(Binar_numpy, connectivity=4)  # centriod (w,h)

    boxes = stats[1:, :]
    points = centroids[1:, :]
    index = (boxes[:, 4] >= min_area)
    boxes = boxes[index]
    points = points[index]
    pre_data = {'num': len(points), 'points': points}
    return pre_data, boxes


def create_model(model_path):
    
    net = Crowd_locator(netName,GPU_ID,pretrained=True)
    net.to(device)
    state_dict = torch.load(model_path, map_location=device)
    if len(GPU_ID.split(','))>1:
        net.load_state_dict(state_dict)
    else:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    net.eval()

    return net



def predict_people_in_frame(img, net): # file_list -> file_path


    if img.mode == 'L':
        img = img.convert('RGB')
    img = img_transform(img)[None, :, :, :]
    slice_h, slice_w = 512,1024
    slice_h, slice_w = slice_h, slice_w

    with torch.no_grad():
        img = Variable(img).to(device)
        b, c, h, w = img.shape
        crop_imgs, crop_dots, crop_masks = [], [], []
        if h * w < slice_h * 2 * slice_w * 2 and h % 16 == 0 and w % 16 == 0:
            [pred_threshold, pred_map, __] = [i.cpu() for i in net(img, mask_gt=None, mode='val')]
        else:
            if h % 16 != 0:
                pad_dims = (0, 0, 0, 16 - h % 16)
                h = (h // 16 + 1) * 16
                img = F.pad(img, pad_dims, "constant")


            if w % 16 != 0:
                pad_dims = (0, 16 - w % 16, 0, 0)
                w = (w // 16 + 1) * 16
                img = F.pad(img, pad_dims, "constant")


            for i in range(0, h, slice_h):
                h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                for j in range(0, w, slice_w):
                    w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)
                    crop_imgs.append(img[:, :, h_start:h_end, w_start:w_end])
                    mask = torch.zeros(1,1,img.size(2), img.size(3)).cpu()
                    mask[:, :, h_start:h_end, w_start:w_end].fill_(1.0)
                    crop_masks.append(mask)
            crop_imgs, crop_masks =  torch.cat(crop_imgs, dim=0), torch.cat(crop_masks, dim=0)

            # forward may need repeatng
            crop_preds, crop_thresholds = [], []
            nz, period = crop_imgs.size(0), 4
            for i in range(0, nz, period):
                [crop_threshold, crop_pred, __] = [i.cpu() for i in net(crop_imgs[i:min(nz, i+period)],mask_gt = None, mode='val')]
                crop_preds.append(crop_pred)
                crop_thresholds.append(crop_threshold)

            crop_preds = torch.cat(crop_preds, dim=0)
            crop_thresholds = torch.cat(crop_thresholds, dim=0)

            # splice them to the original size
            idx = 0
            pred_map = torch.zeros(b, 1, h, w).cpu()
            pred_threshold = torch.zeros(b, 1, h, w).cpu().float()
            for i in range(0, h, slice_h):
                h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                for j in range(0, w, slice_w):
                    w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)
                    pred_map[:, :, h_start:h_end, w_start:w_end] += crop_preds[idx]
                    pred_threshold[:, :, h_start:h_end, w_start:w_end] += crop_thresholds[idx]
                    idx += 1
            mask = crop_masks.sum(dim=0)
            pred_map = (pred_map / mask)
            pred_threshold = (pred_threshold / mask)

        a = torch.ones_like(pred_map)
        b = torch.zeros_like(pred_map)
        binar_map = torch.where(pred_map >= pred_threshold, a, b)

        pred_data, boxes = get_boxInfo_from_Binar_map(binar_map.cpu().numpy())

        return pred_data





if __name__ == '__main__':
    main()




