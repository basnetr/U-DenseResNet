import argparse
import glob
import os
import nibabel as nib
import ntpath
import numpy as np

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def convertprdlabels(inparr):
    '''
        These mappings are taken from IBSR website, labels of freesurfer (cortical labels) that are not defined by IBSR are mapped to graymatter(2)
    '''
    inplbl = [0, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 31, 41, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63, 77]
    outlbl = [0, 3, 1, 1, 3, 2, 2, 2, 2, 2, 1, 1, 3, 2, 2, 1, 2, 2, 0, 3, 1, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 3]
       
    outarr = inparr.copy()
    for idx, inp in enumerate(inplbl):
        outarr[outarr == inp] = outlbl[idx]
    outarr[outarr > 3] = 2 
    return outarr

def dice(im1, im2, tid):
    im1=im1==tid
    im2=im2==tid
    im1=np.asarray(im1).astype(np.bool)
    im2=np.asarray(im2).astype(np.bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    dsc=2. * intersection.sum() / (im1.sum() + im2.sum())
    return dsc

def get_dsc(im1, im2):
    '''
    0=BG, 1=CSF, 2=GM, 3=WM
    '''
    d0 = dice(im1, im2, 0)
    d1 = dice(im1, im2, 1)
    d2 = dice(im1, im2, 2)
    d3 = dice(im1, im2, 3)
    da = np.mean([d1, d2, d3])
    return d0, d1, d2, d3, da

def average_list(lst):
    return sum(lst) / len(lst)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", default='IBSR_nifti_stripped_prediction', help="Path to directory containing subjects", type=str)  
    parser.add_argument("-o", default='IBSR_nifti_stripped_prediction_postprocessed', help="Path to directory containing subjects", type=str)
    parser.add_argument("-gt", default='IBSR_nifti_stripped_conformed', help="Path to directory containing conformed segmentations", type=str)

    args = parser.parse_args()

    prdfiles = glob.glob(os.path.join(args.i, 'IBSR_*.mgz'))

    bgdsc = []
    cfdsc = []
    gmdsc = []
    wmdsc = []
    avdsc = []

    print('bgdsc', 'cfdsc', 'gmdsc', 'wmdsc', 'avdsc')

    for prdfile in prdfiles:

        segfile = os.path.join(args.gt, path_leaf(prdfile).replace('_prd_conformed.mgz', '_seg_conformed.nii.gz'))

        prdnib = nib.load(prdfile)
        prdarr = prdnib.get_fdata()

        # convert 95 labels to 3 labels
        prdarr = convertprdlabels(prdarr)

        segnib = nib.load(segfile)
        segarr = segnib.get_fdata()

        # prdnib = nib.Nifti1Image(prdarr, prdnib.affine, prdnib.header)
        # prdpth = os.path.join(args.o, path_leaf(prdfile).replace('_prd_conformed.mgz', '') + '_prd_conformed_postprocessed.nii.gz')
        # nib.save(prdnib, prdpth)

        dscscores = get_dsc(prdarr, segarr)

        print(path_leaf(prdfile))
        print(dscscores)

        bgdsc.append(dscscores[0])
        cfdsc.append(dscscores[1])
        gmdsc.append(dscscores[2])
        wmdsc.append(dscscores[3])
        avdsc.append(dscscores[4])

    print('Average')
    print(average_list(bgdsc), average_list(cfdsc), average_list(gmdsc), average_list(wmdsc), average_list(avdsc))

if __name__ == '__main__':
    main()