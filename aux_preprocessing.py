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
       var: variable of type pydicom.dataset.FileDataset (specific slice)
       feat: feature whose value we want to retrieve'''
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
        return [val[0:4], val[4:6], val[6:8]]
    if var.data_element(feat).VR=='DS':
        if isinstance(val, int): # single int
            return [int(val)]
        else: # list of ints
            return [int(val_) for val_ in val]
    if var.data_element(feat).VR=='DT':
        return [val[0:4], val[4:6], val[6:8], val[8:10], val[10:12], val[12:14],
                val[15:21]]
    if var.data_element(feat).VR=='FD':
        if isinstance(val, float): # single float
            return [float(val)]
        else: # list of floats
            return [float(val_) for val_ in val]
    if var.data_element(feat).VR=='FL':
        if isinstance(val, float): # single float
            return [float(val)]
        else: # list of floats
            return [float(val_) for val_ in val]
    if var.data_element(feat).VR=='IS':
        if isinstance(val, int): # single int
            return [int(val)]
        else: # list of ints
            return [int(val_) for val_ in val]
    if var.data_element(feat).VR=='TM':
        return [val[0:2], val[2:4], val[4:6], val[5:11]]
    if var.data_element(feat).VR=='US':
        if isinstance(val, int): # single int
            return [int(val)]
        else: # list of ints
            return [int(val_) for val_ in val]

    # String features:
    if (var.data_element(feat).VR=='CS') or (var.data_element(feat).VR=='LO') or (var.data_element(feat).VR=='SH'):
        if feat=='ConvolutionKernel':
            if var.data_element(feat).value=='I40f':
                return [0]
            if var.data_element(feat).value=='I26f':
                return [1]
            else:
                print('ERROR: ConvolutionKernel has unrecognized value.')
        if feat=='PatientSex':
            if var.data_element(feat).value=='F':
                return [0]
            if var.data_element(feat).value=='I26f':
                return [1]
            else:
                print('ERROR: PatientSex has unrecognized value.')
        else:
            print('Not implemented')

    # Special features:
    if var.data_element(feat).VR=='OW':
        print('Not implemented')
    if var.data_element(feat).VR=='SQ':
        print('Not implemented')
        #seq = []
        #for val_ in val:
        #    seq = seq+[get_feature_value(val_,)]
    if var.data_element(feat).VR=='UI':
        print('Not implemented')



    # First, we cast the values of the features that are not scalars
    # into different scalar spaces:
    # if feat=='SpecificCharacterSet':
    #    print('Not implemented')
    # if feat=='ImageType':
    #    print('Not implemented')
    # if feat=='SOPClassUID':
    #    print('Not implemented')
    # if feat=='SOPInstanceUID':
    #    print('Not implemented')
    # if feat=='AcquisitionDateTime':
    #    print('Not implemented')
    # if feat=='SpecificCharacterSet':
    #    print('Not implemented')
    # if feat=='SpecificCharacterSet':
    #    print('Not implemented')

    # The rest of features' values remain the same:
    # else:
    #    var.data_element(feat).value
