from PIL import Image
from transformers import GPT2TokenizerFast
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  
#-------------------------------------------
#These are all my handwritten comments 
#--------------------------------------------
class Dataset: #Defining the dataset class

    def __init__(self, df, tfms): # making the constructor for intialization

        self.df = df
        self.tfms = tfms

    def __len__(self): # Returns length of the dataset

        return len(self.df)

    def __getitem__(self,idx): # Function to transform and return an item from the dataset

        sample = self.df.iloc[idx,:]  # Choosing the image at index = idx
        image = sample['image'] # Extracting the same 
        caption = sample['caption'] 
        image = Image.open(image).convert('RGB')
        image = np.array(image) # the image converts into a array of nums according to pixel intensity 
        augs = self.tfms(image=image) # applies image transformations using the albumentations library.
        image = augs['image'] # albumentations requires you to pass data as a dictionary (image=image) and returns a dictionary,extract the processed image back out using augs['image'].
        caption = f"{caption}<|endoftext|>"
        input_ids = tokenizer(   #tokenizes the captions
            caption,
            truncation=True)['input_ids']
        labels = input_ids.copy() 
        labels[:-1] = input_ids[1:]  
        return image,input_ids,labels
    

def collate_fn(batch):
    image = [i[0] for i in batch]  #these lines of code are using list comprehension to unpack and separate a batch of data 
    input_ids = [i[1] for i in batch]
    labels = [i[2] for i in batch]
    image = torch.stack(image,dim=0) #dim 0 m ek aur add hua by stacking image,inputids,labels
    # Step 1: Run the function and save the resulting dictionary to a temporary variable
    padded_output = tokenizer.pad(
        {'input_ids': input_ids},
        padding='longest',
        return_attention_mask=False,
        return_tensors='pt'
    )

    # Step 2: Extract only the 'input_ids' tensor from that dictionary
    input_ids = padded_output['input_ids']
    labels = tokenizer.pad(
        {'input_ids':labels},
        padding='longest',
        return_attention_mask=False,
        return_tensors='pt'
    )['input_ids']
    #PyTorch's loss function (CrossEntropyLoss—which calculates how "wrong" model is during training) having a hardcoded rule: completely ignore any target labeled as -100.
    mask = (input_ids != tokenizer.pad_token_id).long()
    labels[mask==0]=-100
    return image, input_ids, labels


sample_tfms = [
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.ColorJitter(),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=45, p=0.5),
    A.HueSaturationValue(p=0.3),
]
train_tfms = A.Compose([
    *sample_tfms,
    A.Resize(224,224),
    A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
    ToTensorV2()
])
valid_tfms = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
    ToTensorV2()
])