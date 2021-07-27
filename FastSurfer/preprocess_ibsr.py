'''
Navigate to FastSurferCNN directory after cloning fastsurfer and copy this file

Run Command:

python preprocess_ibsr.py -i IBSR_nifti_stripped -o IBSR_nifti_stripped_conformed

The output preprocessed files are conformed as per fastsurfer's requirements

'''

from data_loader.conform import conform
import argparse
import glob
import os
import ntpath
import nibabel as nib

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", default='IBSR_nifti_stripped', help="Path to directory containing subjects", type=str)  
    parser.add_argument("-o", default='IBSR_nifti_stripped_preprocessed', help="Path to directory containing subjects", type=str)

    args = parser.parse_args()

    subjectdirs = glob.glob(os.path.join(args.i, 'IBSR_*'))

    os.makedirs(args.o, exist_ok=True)

    for subjectdir in subjectdirs:

        print(f'Processing subject {subjectdir}')
        # obtain stripped T1 image
        imgpth = os.path.join(subjectdir, path_leaf(subjectdir) + '_ana_strip.nii.gz')

        # obtain corresponding ground truth segmentation
        segpth = os.path.join(subjectdir, path_leaf(subjectdir) + '_segTRI_ana.nii.gz')

        imgnib = nib.load(imgpth)
        segnib = nib.load(segpth)

        # load data and convert shape from (256, 128, 256, 1) to (256, 128, 256)
        imgarr = imgnib.get_fdata().squeeze() 
        segarr = segnib.get_fdata().squeeze()

        # copies and adapts the passed header to the new image data shape, and affine.
        imgnib = nib.Nifti1Image(imgarr, imgnib.affine, imgnib.header) 
        segnib = nib.Nifti1Image(segarr, segnib.affine, segnib.header) 

        # conforming image and segmentation mask
        imgnib = conform(imgnib, 1)
        segnib = conform(segnib, 0)

        imgpth = os.path.join(args.o, path_leaf(subjectdir) + '_img_conformed.nii.gz')
        segpth = os.path.join(args.o, path_leaf(subjectdir) + '_seg_conformed.nii.gz')
        nib.save(imgnib, imgpth)
        nib.save(segnib, segpth)

        print(f'Image saved as {imgpth}')
        print(f'Mask saved as {segpth}')

if __name__ == '__main__':
    main()