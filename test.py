from config import *
import time
from loss_func import dice
import SimpleITK as sitk
import glob

op_dir = './iSeg-2019-Testing-labels'

def read_med_image(file_path, dtype):
    img_stk = sitk.ReadImage(file_path)
    img_np = sitk.GetArrayFromImage(img_stk)
    img_np = img_np.astype(dtype)
    return img_np, img_stk

def convert_label(label_img):
    label_processed=np.zeros(label_img.shape[0:]).astype(np.uint8)
    for i in range(label_img.shape[2]):
        label_slice=label_img[:, :, i]
        label_slice[label_slice == 10] = 1
        label_slice[label_slice == 150] = 2
        label_slice[label_slice == 250] = 3
        label_processed[:, :, i]=label_slice
    return label_processed

def convert_label_submit(label_img):
    label_processed=np.zeros(label_img.shape[0:]).astype(np.uint8)
    for i in range(label_img.shape[2]):
        label_slice=label_img[:, :, i]
        label_slice[label_slice == 1] = 10
        label_slice[label_slice == 2] = 150
        label_slice[label_slice == 3] = 250
        label_processed[:, :, i]=label_slice
    return label_processed

def get_seg(net, op_dir):
    xstep = 16
    ystep = 16
    zstep = 16
    root_path = './iSeg-2019-Testing'

    for subject_id in range(24, 40):
        time_start = time.perf_counter()
        sub = 'subject-%d-'  % subject_id
        ft1 = os.path.join(root_path, sub + 'T1.hdr')
        ft2 = os.path.join(root_path, sub + 'T2.hdr')

        imT1, imT1_itk = read_med_image(ft1, dtype=np.float32) 
        imT2, imT2_itk = read_med_image(ft2, dtype=np.float32)

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

        time_elapsed = (time.perf_counter() - time_start)
        f= open("output_time.txt","a+")
        f.write("Subject_%d %.16f\n" % (subject_id, time_elapsed))
        print("Subject_%d %.16f\n" % (subject_id, time_elapsed))

        f_pred = os.path.join( op_dir, "subject-%d-label.hdr"  % subject_id )
        whole_pred = (imT1 != 0) * whole_pred
        whole_pred = convert_label_submit(whole_pred)
        whole_pred_itk = sitk.GetImageFromArray(whole_pred.astype(np.uint8))
        whole_pred_itk.SetSpacing(imT1_itk.GetSpacing())
        whole_pred_itk.SetDirection(imT1_itk.GetDirection())
        sitk.WriteImage(whole_pred_itk, f_pred)

if __name__ == '__main__':
    if not os.path.exists(op_dir):
        os.makedirs(op_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DenseResNet(num_init_features=32, growth_rate=16, block_config=(4, 4, 4, 4),num_classes=4).to(device)
    model = 9600
    model = str(model).zfill(5)
    saved_state_dict = torch.load( './checkpoints/model_epoch_'+ model +'.pth' )
    net.load_state_dict(saved_state_dict)
    net.eval()
    
    d = get_seg(net, op_dir)
