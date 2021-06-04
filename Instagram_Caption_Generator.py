import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pytesseract as tess
tess.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'
import sys
import csv
import re
from PIL import Image

#### Loading .csv file containing image locations and captions ####
data = pd.read_csv("captions_csv.csv")

#### Preprocessing data ####

# Function for Translation of slang to normal text
def translator(user_string):
    user_string = user_string.split(" ")
    j = 0
    for _str in user_string:
        # File path which consists of Abbreviations.
        fileName = "slang.txt"
        # File Access mode [Read Mode]
        accessMode = "r"
        with open(fileName, accessMode) as myCSVfile:
            # Reading file as CSV with delimiter as "=", so that abbreviation are stored in row[0] and phrases in row[1]
            dataFromFile = csv.reader(myCSVfile, delimiter="=")
            # Removing Special Characters.
            _str = re.sub('[^a-zA-Z0-9-_.]', '', _str)
            for row in dataFromFile:
                # Check if selected word matches short forms[LHS] in text file.
                if _str.upper() == row[0]:
                    # If match found replace it with its appropriate phrase in text file.
                    user_string[j] = row[1]
            myCSVfile.close()
        j = j + 1
    # Replacing commas with spaces for final output.
    return(' '.join(user_string))

# Removing the record having more than 1 word in the image
text_in_images = []
for i in data['Image File']:
        img = Image.open(i+'.jpg')
        text = tess.image_to_string(img)
        text_in_images.append(text)
del_row = 0
for j in text_in_images:
    x = j.strip().replace('\n'," ").replace('\x0c',"").strip()
    if len(x.split()) > 2:
        data.drop(del_row, axis=0, inplace=True)
    del_row = del_row + 1
data = data.reset_index()       
data.drop(['index'], axis=1, inplace=True)

# Removing images having non string captions, no captions, 1 word captions or just emojis as caption
c=0
for k in data['Caption']:
    if type(k)!=str or len(k.split())<2:
        data.drop(c, axis=0, inplace=True)
    c=c+1 
data = data.reset_index()       
data.drop(['index'], axis=1, inplace=True)    

# Slang translation; Removing all characters except English, Punctuations and Emojis & lower casing, and creating tokens by considering punctuations and emojis as separate words
caption_data = data.copy()
for r, p in enumerate(caption_data['Caption']):
    f=""
    for q in p:
        if ((ord(q)<127 and ord(q)>31) or (ord(q)>120000 and ord(q)<130000)):
            f=f+q
    caption_data['Caption'][r] = f
for r, l in enumerate(caption_data['Caption']):
    f=""
    for o in l:
        if (o in '''!()-[]{};:'"\,<>./?@#$%^&*_~”\n“–=''') or (ord(o)>120000 and ord(o)<130000):
            f = f+" "+o+" "
        else:
            f = f+o           
    lower_text = translator(f).lower()
    caption_data['Caption'][r] = lower_text.split()  

#### Hyperparameter initialization ####
num_training = int(0.7*(data.shape[0]))
num_validation = int(0.2*(data.shape[0]))
num_testing = int(0.1*(data.shape[0]))
batch_size=200

#### Prepare the training, validation and test splits ####
mask = list(range(num_training))
train_dataset = torch.utils.data.Subset(data, mask)
mask = list(range(num_training, num_training + num_validation))
val_dataset = torch.utils.data.Subset(data, mask)
mask = list(range(num_training, num_training + num_validation + num_testing))
test_dataset = torch.utils.data.Subset(data, mask)

#### Data loaders ####
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)