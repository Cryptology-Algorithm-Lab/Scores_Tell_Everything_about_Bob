import torch
import boto3
import numpy as np

ACCESS_KEY = "YOUR ACCESS_KEY"
SECRET_KEY = "YOUR SECRET_KEY"
REGION = "YOUR REGION"

# Uncomment to create only one client
# client = boto3.client("rekognition", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=REGION)

def sigmoid_inv(y, a, x0, k , b):

    x = -1/k * np.log(a/(y-b) - 1) + x0

    return x

def cal_inv(param : list, confidence_float):

    return sigmoid_inv(confidence_float, *param)

def cosine_score_AWS(target_id):
    client = boto3.client("rekognition", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=REGION)
    confidence_score = torch.zeros(99)
    estimated_cosine_score = torch.zeros(99)
    for i in range(99):
        if i==0:
            print(" ")
            print("Querying to AWS...")
        ofs = './sample_OFS/'+str(i)+'.png'
        target = './sample_LFW/L/'+str(target_id)+'.png'

        imageSource = open(ofs, 'rb')
        imageTarget = open(target, 'rb')

        try:

            response = client.compare_faces(SimilarityThreshold=0,
                                        SourceImage={'Bytes': imageSource.read()},
                                        TargetImage={'Bytes': imageTarget.read()})

            confidence_score[i] = response['FaceMatches'][0]['Similarity']
            # transformation to cosine similarity score from confidence score
            estimated_cosine_score[i] = 1 - cal_inv(np.array([ 9.9e-01,  7.5e-01, -2.5e+01,  4.0e-05]) , 0.01*confidence_score[i])
            #print(estimated_cosine_score[i])
        except:

            print("Error with detection")

    return estimated_cosine_score

def query_AWS(target_id,recon,type12):
    client = boto3.client("rekognition", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=REGION)
    if type12==1:
        target = './sample_LFW/L/'+str(target_id)+'.png'
    else:
        target = './sample_LFW/R/'+str(target_id)+'.png'

    imageSource = open(recon, 'rb')
    imageTarget = open(target, 'rb')

    response = client.compare_faces(SimilarityThreshold=0,
                                        SourceImage={'Bytes': imageSource.read()},
                                        TargetImage={'Bytes': imageTarget.read()})

    return response['FaceMatches'][0]['Similarity']
