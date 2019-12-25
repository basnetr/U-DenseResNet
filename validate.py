from config import *
import time
from loss_func import dice
import SimpleITK as sitk
import glob

def convert_label(label_img):
    '''
    function that converts 0, 10, 150, 250 to 0, 1, 2, 3 labels for BG, CSF, GM and WM
    '''
    label_processed = np.where(label_img==10, 1, label_img)
    label_processed = np.where(label_processed==150, 2, label_processed)
    label_processed = np.where(label_processed==250, 3, label_processed)
    return label_processed

def read_med_image(file_path, dtype):
    img_stk = sitk.ReadImage(file_path)
    img_np = sitk.GetArrayFromImage(img_stk)
    img_np = img_np.astype(dtype)
    return img_np, img_stk

def predict(net):
    xstep = 16
    ystep = 16
    zstep = 16
    root_path = './iSeg-2017-Training'

    sub = 'subject-9-'
    ft1 = os.path.join(root_path, sub + 'T1.hdr')
    ft2 = os.path.join(root_path, sub + 'T2.hdr')
    fgt = os.path.join(root_path, sub + 'label.hdr')

    imT1, imT1_itk = read_med_image(ft1, dtype=np.float32) 
    imT2, imT2_itk = read_med_image(ft2, dtype=np.float32)
    imGT, imGT_itk = read_med_image(fgt, dtype=np.uint8)

    imGT = convert_label(imGT)
    # print(imGT.shape)
    mask = imT1 > 0
    mask = mask.astype(np.bool)

    imT1_norm = (imT1 - imT1[mask].mean()) / imT1[mask].std()
    imT2_norm = (imT2 - imT2[mask].mean()) / imT2[mask].std()

    input1 = imT1_norm[:, :, :, None]
    input2 = imT2_norm[:, :, :, None]

    inputs = np.concatenate((input1, input2), axis=3)
    inputs = inputs[None, :, :, :, :]
    image = inputs.transpose(0, 4, 1, 3, 2)
    image = torch.from_numpy(image).float().to(device)   

    _, _, C, H, W = image.shape
    deep_slices   = np.arange(0, C - crop_size[0] + xstep, xstep)
    height_slices = np.arange(0, H - crop_size[1] + ystep, ystep)
    width_slices  = np.arange(0, W - crop_size[2] + zstep, zstep)

    whole_pred = np.zeros((1,)+(num_classes,) + image.shape[2:])
    count_used = np.zeros((image.shape[2], image.shape[3], image.shape[4])) + 1e-5

    with torch.no_grad():
        for i in range(len(deep_slices)):
            for j in range(len(height_slices)):
                for k in range(len(width_slices)):
                    deep = deep_slices[i]
                    height = height_slices[j]
                    width = width_slices[k]
                    image_crop = image[:, :, deep   : deep   + crop_size[0],
                                                height : height + crop_size[1],
                                                width  : width  + crop_size[2]]

                    outputs = net(image_crop)
                    whole_pred[slice(None), slice(None), deep: deep + crop_size[0],
                                height: height + crop_size[1],
                                width: width + crop_size[2]] += outputs.data.cpu().numpy()

                    count_used[deep: deep + crop_size[0],
                                height: height + crop_size[1],
                                width: width + crop_size[2]] += 1

    whole_pred = whole_pred / count_used
    whole_pred = whole_pred[0, :, :, :, :]
    whole_pred = np.argmax(whole_pred, axis=0)
    
    whole_pred = whole_pred.transpose(0,2,1)
    whole_pred = (imT1 != 0) * whole_pred
    d0 = dice(whole_pred, imGT, 0)
    d1 = dice(whole_pred, imGT, 1)
    d2 = dice(whole_pred, imGT, 2)
    d3 = dice(whole_pred, imGT, 3)
    da = np.mean([d1, d2, d3])
    
    return [round(d0*100,2), round(d1*100,2), round(d2*100,2), round(d3*100,2), round(da*100,2)]

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DenseResNet(num_init_features=32, growth_rate=16, block_config=(4, 4, 4, 4),num_classes=4).to(device)

    f1 = open('output.txt', 'a+')
    model = 6000
    model = str(model).zfill(5)
    saved_state_dict = torch.load( './checkpoints/model_epoch_'+ model +'.pth' )
    net.load_state_dict(saved_state_dict)
    net.eval()

    d = predict(net)
    
    print( '%s %2.2f %2.2f %2.2f %2.2f %2.2f\n' % (model, d[0], d[1], d[2], d[3], d[4]) )
