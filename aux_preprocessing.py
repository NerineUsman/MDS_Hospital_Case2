'''Functions used for preprocessing of the raw data from the hospital'''
import os
import numpy as np
from matplotlib import pyplot as plt

from os.path import dirname, join
from pprint import pprint

import pydicom
from pydicom.data import get_testdata_files
from pydicom.filereader import read_dicomdir


def _getstudies(filepath):
    filepath = filepath
    dicom_dir = read_dicomdir(filepath)
    base_dir = dirname(filepath)

    allstudies = {}

    # go through the patient record and print information
    for patient_record in dicom_dir.patient_records:
        #if (hasattr(patient_record, 'PatientID') and hasattr(patient_record, 'PatientName')):
            #print("Patient: {}: {}".format(patient_record.PatientID,
            #                               patient_record.PatientName))
        studies = patient_record.children
        # got through each serie
        for study in studies:
            #print(" " * 4 + "Study {}: {}: {}".format(study.StudyID,
            #                                          study.StudyDate,
            #                                          study.StudyDescription))
            allstudies[study.StudyID] = []
            all_series = study.children
            # go through each serie
            tmpseries = {}
            for series in all_series:
                image_count = len(series.children)
                plural = ('', 's')[image_count > 1]

                # Write basic series info and image count

                # Put N/A in if no Series Description
                if 'SeriesDescription' not in series:
                    series.SeriesDescription = "N/A"
                #print(" " * 8 + "Series {}: {}: {} ({} image{})".format(
                #    series.SeriesNumber, series.Modality, series.SeriesDescription,
                #    image_count, plural))

                # Open and read something from each image, for demonstration
                # purposes. For simple quick overview of DICOMDIR, leave the
                # following out
                #print(" " * 12 + "Reading images...")
                image_records = series.children
                image_filenames = [join(base_dir, *image_rec.ReferencedFileID)
                                   for image_rec in image_records]

                datasets = [pydicom.dcmread(image_filename)
                            for image_filename in image_filenames]

                patient_names = set(ds.PatientName for ds in datasets)
                patient_IDs = set(ds.PatientID for ds in datasets)

                # List the image filenames
                #print("\n" + " " * 12 + "Image filenames:")
                #print(" " * 12, end=' ')
                #pprint(image_filenames, indent=12)

                # Expect all images to have same patient name, id
                # Show the set of all names, IDs found (should each have one)
                #print(" " * 12 + "Patient Names in images..: {}".format(
                #    patient_names))
                #print(" " * 12 + "Patient IDs in images..: {}".format(
                #    patient_IDs))
                tmpseries[int(series.SeriesNumber)] = datasets

            allstudies[study.StudyID].append(tmpseries)
    return allstudies

def check_len1(my_dict):
    '''Checks if all the values of the dictionary are single-element lists
       ---Inputs---
       my_dict: dictionary with lists as values'''
    a = 0
    for key in list(my_dict.keys()):
       if len(my_dict[key]) != 1:
           print('ERROR: %s has %d elements'%(key,len(my_dict[key])))
           a = 1
    if a == 0:
       print('Everything ok!')

def get_feature_value(var,feat):
    '''Gets the value of a feature of a particular slice. All the values are
    casted into real numbers.
       ---Inputs---
       var: variable of type pydicom.dataset.FileDataset (slice)
       feat: feature whose value we want to retrieve
       ---Outputs---
       Returns (multi- or uni-dimensional) feature as a list.'''
    val = var.data_element(feat).value

    # Numerical features:
    if var.data_element(feat).VR=='AS':
        units = val[-1]
        if units=='Y':
            return [float(val[0:3])]
        if units=='M':
            return [float(val[0:3])/12]
        if units=='W':
            return [float(val[0:3])*7/365]
        if units=='D':
            return [float(val[0:3])/365]
    if var.data_element(feat).VR=='DA':
        if feat=='DateOfLastCalibration':
            calibdate = int(val[4:6])*30+int(val[6:8])
            acval = var.data_element('AcquisitionDate').value
            actime = int(acval[4:6])*30+int(acval[6:8])
            return [actime-calibdate]
        else:
            return [int(val[4:6])*30+int(val[6:8])]
    if var.data_element(feat).VR=='DS':
        if isinstance(val, pydicom.multival.MultiValue): # list of ints
            val = [int(val_) for val_ in val]
        else: # single int
            val = [int(val), 1] # The DS elements are floats expressed as a
                                 # ratio of two ints. If the number is a single
                                 # int, it corresponds to that number divided
                                 # by 1.
        if feat in ['ContrastBolusTotalDose','ContrastBolusVolume','ContrastFlowDuration','ContrastFlowRate']:
            if val in [[125, 1],[30, 14],[3, 2]]:
                return [0]
            if val in [[100, 1],[25, 1],[3, 1]]:
                return [1]
            else:
                print('ERROR: '+feat+' has unrecognized value.')
        elif feat in ['KVP','WindowCenter','WindowWidth']:
            if val in [[100, 1],[60, -500],[375, 1500]]:
                return [0]
            if val in [[120, 1],[50, -500],[350, 1500]]:
                return [1]
            else:
                print('ERROR: '+feat+' has unrecognized value:')
        elif feat in ['ReconstructionDiameter', 'SliceLocation', 'TableHeight']:
            return [val[0]]
        else:
            return val
    if var.data_element(feat).VR=='DT':
        return [val[0:4], val[4:6], val[6:8], val[8:10], val[10:12], val[12:14],
                val[15:21]]
    if var.data_element(feat).VR=='FD':
        if feat=='DataCollectionCenterPatient':
            vall = [float(val_) for val_ in val]
            return vall[1:]
        if isinstance(val, list): # list of floats
            return [float(val_) for val_ in val]
        else: # single float
            return [float(val)]
    if var.data_element(feat).VR=='FL':
        if feat=='CalciumScoringMassFactorDevice':
            if var.data_element(feat).value==[0.6430000066757202, 0.6710000038146973, 0.6980000138282776]:
                return [0]
            if var.data_element(feat).value==[0.7429999709129333, 0.7789999842643738, 0.8119999766349792]:
                return [1]
            else:
                print('ERROR: CalciumScoringMassFactorDevice has unrecognized value.')
        if isinstance(val, list): # list of floats
            return [float(val_) for val_ in val]
        else: # single float
            return [float(val)]
    if var.data_element(feat).VR=='IS':
        if isinstance(val, list): # list of ints
            return [int(val_) for val_ in val]
        else: # single int
            return [int(val)]
    if var.data_element(feat).VR=='TM':
        if feat=='ContrastBolusStartTime':
            starttime = float(val[4:6])+float(val[2:4])*60+float(val[0:2])*3600
            stopval = var.data_element('ContrastBolusStopTime').value
            stoptime = float(stopval[4:6])+float(stopval[2:4])*60+float(stopval[0:2])*3600
            if int(stoptime-starttime)==74:
                return [0]
            elif int(stoptime-starttime)==26:
                return [1]
        elif feat=='ContrastBolusStopTime':
            print('ContrastBolusStopTime should be ignored (not implemented).')
        elif feat=='TimeOfLastCalibration':
            starttime = float(val[2:4])/60+float(val[0:2])
            stopval = var.data_element('AcquisitionTime').value
            stoptime = float(stopval[2:4])/60+float(stopval[0:2])
            return [stoptime-starttime]
        elif feat=='ContentTime':
            starttime = float(val[4:6])+float(val[2:4])*60
            stopval = var.data_element('AcquisitionTime').value
            stoptime = float(stopval[4:6])+float(stopval[2:4])*60
            return [starttime-stoptime]
        else:
            return [float(val[0:2])+float(val[2:4])/60]
    if var.data_element(feat).VR=='US':
        if isinstance(val, list): # list of ints
            return [int(val_) for val_ in val]
        else: # single int
            return [int(val)]

    # String features:
    if (var.data_element(feat).VR=='CS') or (var.data_element(feat).VR=='LO') or (var.data_element(feat).VR=='SH'):
        if feat=='ConvolutionKernel':
            if var.data_element(feat).value[0]=='I40f':
                return [0]
            if var.data_element(feat).value[0]=='I26f':
                return [1]
            else:
                print('ERROR: ConvolutionKernel has unrecognized value.')
        if feat=='PatientSex':
            if var.data_element(feat).value=='F':
                return [0]
            if var.data_element(feat).value=='M':
                return [1]
            else:
                print('ERROR: PatientSex has unrecognized value.')
        else:
            print('String feature ignored.')

    # Sequence features:
    if var.data_element(feat).VR=='SQ':
        seq = var.data_element(feat)[0]

        if (feat=='ProcedureCodeSequence') or (feat=='RequestedProcedureCodeSequence'):
            if seq.data_element('CodeValue').value=='C5-05':
                return [0]
            if seq.data_element('CodeValue').value=='C5-01':
                return [1]
            else:
                print('ERROR: CodeValue subfeature has unrecognized value.')
        else:
            print('Sequence feature ignored.')

    # Special features:
    if var.data_element(feat).VR=='OW':
        print('Not implemented')

    if var.data_element(feat).VR=='UI':
        print('UID feature ignored.')

def get_feature_value_numerSection(var,feat):
    '''Gets the value of a feature of a particular slice. All the values are
    casted into real numbers. THIS IS A TEST FUNCTION THAT IS ONLY USED IN
    THE preprocessing_CT.ipynb FILE. DO NOT USE.
       ---Inputs---
       var: variable of type pydicom.dataset.FileDataset (slice)
       feat: feature whose value we want to retrieve
       ---Outputs---
       Returns (multi- or uni-dimensional) feature as a list.'''
    val = var.data_element(feat).value

    # Numerical features:
    if var.data_element(feat).VR=='AS':
        units = val[-1]
        if units=='Y':
            return [float(val[0:3])]
        if units=='M':
            return [float(val[0:3])/12]
        if units=='W':
            return [float(val[0:3])*7/365]
        if units=='D':
            return [float(val[0:3])/365]
    if var.data_element(feat).VR=='DA':
        if feat=='DateOfLastCalibration':
            calibdate = int(val[4:6])*30+int(val[6:8])
            acval = var.data_element('AcquisitionDate').value
            actime = int(acval[4:6])*30+int(acval[6:8])
            return [actime-calibdate]
        else:
            return [int(val[4:6])*30+int(val[6:8])]
    if var.data_element(feat).VR=='DS':
        if isinstance(val, pydicom.multival.MultiValue): # list of ints
            val = [int(val_) for val_ in val]
        else: # single int
            val = [int(val), 1] # The DS elements are floats expressed as a
                                 # ratio of two ints. If the number is a single
                                 # int, it corresponds to that number divided
                                 # by 1.
        return val
    if var.data_element(feat).VR=='DT':
        return [val[0:4], val[4:6], val[6:8], val[8:10], val[10:12], val[12:14],
                val[15:21]]
    if var.data_element(feat).VR=='FD':
        if isinstance(val, list): # list of floats
            return [float(val_) for val_ in val]
        else: # single float
            return [float(val)]
    if var.data_element(feat).VR=='FL':
        if isinstance(val, list): # list of floats
            return [float(val_) for val_ in val]
        else: # single float
            return [float(val)]
    if var.data_element(feat).VR=='IS':
        if isinstance(val, list): # list of ints
            return [int(val_) for val_ in val]
        else: # single int
            return [int(val)]
    if var.data_element(feat).VR=='TM':
        return [float(val[0:2])+float(val[2:4])/60]
    if var.data_element(feat).VR=='US':
        if isinstance(val, list): # list of ints
            return [int(val_) for val_ in val]
        else: # single int
            return [int(val)]

    # String features:
    if (var.data_element(feat).VR=='CS') or (var.data_element(feat).VR=='LO') or (var.data_element(feat).VR=='SH'):
        if feat=='ConvolutionKernel':
            if var.data_element(feat).value[0]=='I40f':
                return [0]
            if var.data_element(feat).value[0]=='I26f':
                return [1]
            else:
                print('ERROR: ConvolutionKernel has unrecognized value.')
        if feat=='PatientSex':
            if var.data_element(feat).value=='F':
                return [0]
            if var.data_element(feat).value=='M':
                return [1]
            else:
                print('ERROR: PatientSex has unrecognized value.')
        else:
            print('String feature ignored.')

    # Sequence features:
    if var.data_element(feat).VR=='SQ':
        seq = var.data_element(feat)[0]

        if (feat=='ProcedureCodeSequence') or (feat=='RequestedProcedureCodeSequence'):
            if seq.data_element('CodeValue').value=='C5-05':
                return [0]
            if seq.data_element('CodeValue').value=='C5-01':
                return [1]
            else:
                print('ERROR: CodeValue subfeature has unrecognized value.')
        else:
            print('Sequence feature ignored.')

    # Special features:
    if var.data_element(feat).VR=='OW':
        print('Not implemented')

    if var.data_element(feat).VR=='UI':
        print('UID feature ignored.')

def low_rank_C(u,s,v,k):
    '''Reduces the rank of a matrix C which is decomposed using SVD as
       C = u*np.diag(s)*v, i.e. u, s, v = np.linalg.svd(C).
        --- Inputs ---
            u: term matrix (WxW array, float)
            s: standard values of C (Tx1 array, str)
            v: transpose document matrix (TxT array, float)
            k: reduced rank of the new C (int)
        --- Outputs ---
            uk: k first columns of u (Wxk array, str)
            sk: first k standard values (kx1 array, str)
            vk: k first columns of u (kxT array, str)'''
    T = len(s)
    #sk = np.concatenate((s[0:k],np.array([0]*(T-k))),axis=None)
    sk = s[:k]
    uk = u[:,:k]
    vk = v[:k,:]
    # Ck = np.matmul(np.matmul(uk,np.diag(sk)),vk)
    return uk, sk, vk

def sizeSlice(s):
    ## determines the volume of a slice in dm^3
    threshold_mask = s.pixel_array > 800
    tmpmask = ndimage.binary_erosion(threshold_mask,iterations =6)
    closedmask = ndimage.binary_fill_holes(tmpmask)
    n_pixels = np.sum(closedmask)      ## # pixels above threshold
    v_pixel = s.PixelSpacing[0]*s.PixelSpacing[1]*s.SliceThickness *10**(-6) # volume 1 pixel in leters
    return n_pixels*v_pixel

def sizePatient(patient):
    ## gives volume of scanned body in dm^3
    volume = 0
    for s in patient:
        volume += sizeSlice(s)
    return volume

