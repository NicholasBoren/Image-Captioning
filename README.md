# Automatic Generation of Image Captions
Group: Anna Batra & Nicholas Boren

## Abstract

In this project we work on image to text with automatic generation of image captions, using a CNN-RNN neural network. Building off of pre-existing code, we continue to experiment and try to improve upon the model by updating and adding in new components to the pre-existing code.

These new components include adding on character level training and using bi-LSTMs with mixed character and word level training. We then compare the three models, word-level, character-level, and mixed word and character level, to each other.

\<Insert Example image>

\<Insert Example Caption Output from our Model>

## Video
Here is a video explaining our project (the same information stated below). We also have a live demo at the end with our finished model generating captions for a few images.

## Problem Statement

We are exploring whether a two-layer bi-LSTM with mixed character and word level training model will outperform a single-layer pure word-level LSTM model and a single layer pure character-level LSTM model.

As we do this, we will also explore if the single-layer pure word-level LSTM model will outperform the character-level one.

## Starter Code and Datasets
We build off of the starter code provided by the [Udacity Computer Vision Nanodegree Program](https://github.com/sauravraghuvanshi/Udacity-Computer-Vision-Nanodegree-Program/tree/master/project_2_image_captioning_project). They use a CNN-RNN architecture model for automatically generating image captions. The network is originally trained on the Microsoft Common Objects in COntext (MS COCO) 2014 dataset. 

We trained, validated, and tested on the [Microsoft Common Objects in COntext (MS COCO)](https://cocodataset.org/#home) 2017 images dataset. MS COCO is one of the public datasets most commonly used for image captioning.

We also made use of the [Sama-Coco Dataset](https://www.sama.com/sama-coco-dataset/), a relabelling of the Coco-2017 dataset, for the annotations. The annotations are much more precise and comphrehensive than the original, and there are also notably more than one caption per image. This better reflects the variability in captions as language is productive. Our hope is that this will better improve the model in terms of training.

## Related Work
The paper [Multilingual Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Models and Auxiliary Loss](https://paperswithcode.com/paper/multilingual-part-of-speech-tagging-with) explores using bi-LSTMs for POS tagging. They use a two-layer bi-LSTM, where the first is character level and the second is both character level and word level. This paper found out that this model does better than just word level. This heavily influenced us in choosing to create a decoder like this, as it may possibly improve our model as well.

## Methodology

Originally, the pre-existing code used the MS COCO 2014 dataset for images and annotations. We improved upon it by using the newer 2017 images dataset and making use of the Sama-Coco Dataset for more precise and comphrehensive annotations.

In the pre-existing code, it makes use of a single layer LSTM, trained on the word-level. For this project, in addition to the Word Vocabulary code given, we created one for the Char Vocabulary. We also altered the DataLoader to take into account the new vocabulary. With our new vocabulary, we were able to train a character-level model as well.

 We will also implement our Decoder with a two-layer bi-LSTM in the same way as the POS tagging paper from our related work section. We will then compare it to our word level and character level models using perplexity as our metric.

## Experiments/Evaluation

### Hyper-parameters

Here are the hyper-parameters we used for our models:
| Hyper-parameters  | Word-Level    | Character-Level   | Mixed Word & Character Level (Bi-LSTMs) | 
| -----------       | -----------   | --                | -----                         |
| Batch Size        | 32            | 10                | s|
| Vocab Threshold   | 6             | 6                 |s|
| Embedding Size    | 512           | 512               |s|
| Hidden Size       | 512           | 512               |s|
| Number of Epochs  | 10            | 1                 |

### Word-Level Model

For the word-level model, we let it run until the end of 3 epochs. The model took about _____ time to run, and we noticed that it actually did it's best in the 2nd epoch so we stopped it from running more.

Here are the results for the word-level model at the end of each epoch. We can see that it does the best at the end of the second epoch with a loss of 1.8976 and a perplexity of 6.66964.

| Word-Level  | Epoch 1    | Epoch 2   | Epoch 3 | 
| ----------- | -----------| --        | -----   |
| Loss        | 2.2183     | 1.8976    | 2.0530  |
| Perplexity  | 9.19188    | 6.66964   | 7.79131 |

Here's a picture 

### Character-Level Model

For the word-level model, we let it run until the end of 3 epochs. The model took about _____ time to run, and we noticed that it actually did it's best in the 2nd epoch so we stopped it from running more.

Here are the results for the word-level model at the end of each epoch. We can see that it does the best at the end of the second epoch with a loss of 1.8976 and a perplexity of 6.66964.

| Word-Level  | Epoch 1    | Epoch 2   | Epoch 3 | 
| ----------- | -----------| --        | -----   |
| Loss        | 2.2183     | 1.8976    | 2.0530  |
| Perplexity  | 9.19188    | 6.66964   | 7.79131 |

Here's a picture 

### Mixed Word & Character-Level Model (Bi-LSTMs)

For the word-level model, we let it run until the end of 3 epochs. The model took about _____ time to run, and we noticed that it actually did it's best in the 2nd epoch so we stopped it from running more.

Here are the results for the word-level model at the end of each epoch. We can see that it does the best at the end of the second epoch with a loss of 1.8976 and a perplexity of 6.66964.

| Word-Level  | Epoch 1    | Epoch 2   | Epoch 3 | 
| ----------- | -----------| --        | -----   |
| Loss        | 2.2183     | 1.8976    | 2.0530  |
| Perplexity  | 9.19188    | 6.66964   | 7.79131 |

Here's a picture 

## Results

We are evaluating using the perplexity metric.

## Future Steps
When we first started this project, there was a lot we envisioned for it but could not accomplish it due to the short timeframe and limited GPU.

The next steps we would like to look into is making use of our validation set. We started altering the DataLoader to take into an account of mode for validation, but did not make far progress due to limited time and re-figuring out goals for this project. Initially, we imagined that we could get through using the CIDEr evaluation metric. Using this along with the validation set, we thought it would be better than perplexity for making improvements in the model based off the hyperparameters. But having done quite a lot of research on this, we found out that adding this onto our plate will take quite a lot of time.

Here are the resources we look into about this:

</list resources>

Our other hope was to create a much better looking website that can also allow the user to submit their own pictures and retrieve captions from each of our three models.


## Github Repository
You can find our [Github repository here](https://github.com/NicholasBoren/Image-Captioning).