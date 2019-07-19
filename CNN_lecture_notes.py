# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 19:02:16 2019

@author: Akshay
"""
"""Convolutional neural networks"""
"""the way computers are going to process the images
input image ------> CNN-------> output label(image class)
It is able to recognize these features by :-
-NN leverages the fact that the image(B/W) is a 2d array of 2X2 pixels; 0<pixel value<=255
 and for coloured images 2x2px is a 3d array of 3(rgb) 2d matrices so by combining all three we can make the final coloured image recognuisez by the nn
-- for a simple smile emoticon on a pixelated display:-
           for all pixel blocks forming the image on the screen , 1 is given to those blocks
           and 0 to the rest of the blocks of pixels


step1:- convolution
step2:- Max pooling
step3:- Flattening
step4:- Full connection

***** convolution*******
(f*g)(t)= integration from -infi to _infi [f(t)g(t-r)dr]
-feature detector or kernel or filter
feature detector purpose is to detect some feature and using it we loses info 
so by applying multiple feature detector of filters we can sharpen or make image better of blur it and so on
-convoultion is to find features into the image then put them into feature map 


****STEP-1(B) RELU (rectifier linear units)******
-we create many feature maps to obtain our first convolutional layer 
-on this we apply our rectifier function
- we apply this to increase non linearity in our image or our convolutional model
- we do this as images are themselves nonlinear
-by doing the filtering we risk that we might make it linear and thus disrupt it ,so we need to make sure it stays nonlinear

********STEP2:- MAX POOLING(down sampling)**********
What is pooling?
ans. we want to make our CNN to recognize all shapes and sizes images of a single entity - spacial invariance(it doesnt care if features are distorted or shaped in some way)
-we take a box of nxn size and then we place it in the top left corner and we find the max value in the box and diregard the other 3 , then we move the box to the right by stride and continue
-this is what max pooling is , keeping the max numbers in the box in the pooled feature map , thus the feature is preserved in the final pooled map.
we are  reducing the number of parameters by 75% and thus preventing overfitting
-avg pooling
-mean pooling

***********STEP3-FLATTENING********
-pooled feature map --------> flatten it into a single column ---> input layer of a future ANN



***********STEP4:- FULL CONNECTION********
-adding an ANN to this CNN i.e input layer, fully connected hidden layers , output layers .

**************Summary*******************
input image----(convolution  )----->convolutional layer(applying RELU)-------(pooling-make sure that we have special invariance in our NN , avoiding overfitting of data)---------> pooling layer--------(flattening)---->input layer of a future ANN

******softmax and cross entropy*******
- softmax function f(z)= e^zj/ sum e^zk
-it brings the values to 0 and 1 and makes sure tht they add upto 1

cross entropy function:-
h(p,q )=-sum p(x)log q(x)
-a cross entropy function is called a loss function
so if dog=0.9 and cat=0.1 then q =dog and p-[1,0]
 
for 2 NN ; NN1 and NN2 
if for 3 observations nn1 outperforms nn2:-
classification error - same for both
mean squared error - nn2 =0.75 and nn1-0.25
cross entropy- nn1=0.38 and nn2=1.06

#part1:-
*convolutional trick is to preserve and structure some images or classify videos and photographs 
*goal- classify images and try to tell for each image their class.
*-we will create a cnn to predict the image is of cat or dog
*-general model can also be applied for medical sciences and other fields
*- we need to do some image preprocessing as we have data now in form of images and not in csv file format

-how can we extract the info of the dependent variable
-we can make separate folders for train and test set
- we can differentiate all the images by diff names
the other solution is using keras:-
-it contains some ways to import the images in a unique way by preparing a special structure for the dataset
-make 2 diff folders for the cats and dogs and the train and test set folders also

- so we dont need to ,import the dataset and no need to encode them and no need to make train and test set, as it is done manualy ---
"""

"""PART1:- building the convolutional neural netowrk"""
from keras.models import Sequential #we can initialize ann  in 2 ways either sequence of layers or a graph
from keras.layers import Convolution2D#convoultion step in which we add convolutional layer to deal with images
from keras.layers import MaxPooling2D#pooling layers
from keras.layers import Flatten#flattening 
from keras.layers import Dense#used to add a fully connected layer to ann

"""Initialising the CNN"""
classifier = Sequential()#as a sequence of layers

"""STEP1:-CONVOLUTION"""
classifier.add(Convolution2D(32, 3, 3 , input_shape=(64,64,3),activation='relu'))#32 feature detectors of 3x3 size, we will convert all our images into a single format so that the size remains same for all
#so we need to specify in which format the images will be converted into
#input images are converted into arrays {for 3d-images red,green,blue}
#256x256 is the format but we will use a smaller format i.e64x64 so the code executes faster

"""STEP2-POOLING"""
#reducing the size of the feature map matrix
classifier.add(MaxPooling2D(pool_size=(2,2)))#we need to fix the size of the subtable that selects the max values


"""line that adds another convolution layer"""
classifier.add(Convolution2D(32, 3, 3 ,activation='relu'))
#applying max pooling on this layer
classifier.add(MaxPooling2D(pool_size=(2,2)))
#if we need more accuracy then add another layer and make it 64 bit instead of 32 in the next one 

"""STEP3-FLATTENING"""
#taking all our pooled feature maps and put them into one single vector  ,this single vector will be the input layer to our ANN
"""2 ques-
1.y dont we lose the special structure by flattening them:-
because by creating our feature maps we extracted the special structure info , the high numbers represent the special structures of the  input image ,we keep this special structure info into this big vector
2.y didnt we directly take all the input images and apply flattening without applying the previous steps , then we dont get any info about the surroundings of the pixel
the high number represents the tiny specific feature that the feature detector can extract from the input image through the convolution operation and therefore we keep this special structure info of the input image
"""
classifier.add(Flatten())#no need to input parameters 

"""STEP4-full connection"""
#now we have our input layer and now we need to create our hidden layer and the output layer
classifier.add(Dense(output_dim=128, activation='relu'))#output-dim-number of nodes in the hidden layers
#adding the output layer
classifier.add(Dense(output_dim=1, activation='sigmoid'))#binary outcome for 2 ;cats and dogs, the oredicted prob of 1 class either cats or dogs - output_dim

"""compiling the CNN"""
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

"""PART2- FITTING THE CNN TO OUR IMAGES"""
"""overfitting occurs when we have very less data and we train our model on it so it tries to find relations between all these data and thus on small data it tries to overfit relations
- it need  to find some patterns on the images
-8k images are not much so we need a trick  of Image Augmentation
:-it will create batches of images without adding any more images and in each batch it will apply some random tranformation like rotating , shifitng, diagonal and so eventually we get many more diverse images and a lot more images to train from
"""
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,#rescale all our pixel values between 0 and 1 
        shear_range=0.2,#to apply random transitons
        zoom_range=0.2,#some random zooms
        horizontal_flip=True)#images will be flipped horizontally

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                'dataset/training_set',
                                                target_size=(64, 64),#size of our images that is expected in our cnn model
                                                batch_size=32,#random size batches in which some random samples will be included
                                                class_mode='binary')#if your class is binary or has more than 2 categories

test_set = test_datagen.flow_from_directory(
                                                'dataset/test_set',
                                                target_size=(64,64 ),
                                                batch_size=32,
                                                class_mode='binary')

classifier.fit_generator(   training_set,
                            steps_per_epoch=50,#numberof images we have in our training set
                            epochs=25,
                            validation_data = test_set,
                            validation_steps=2000)#no. of images in our test set

#now if we can achieve accuracy of more than 80% ?
#add a convolution layer as well as a fully connected layer
#85% for training set and 82% for the test set after the second convolutional layer








