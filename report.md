# VQA 
* Authors: Rafael Garcia, Bernat Joseph, Pau Gil, Jordi Suñer
* Team Advisor: Issey Masuda
* November 2020


# Introduction
Visual Question Answering (VQA) it's aiming to answer Free-form and open-ended Question about an Image, using Computer Vision & Language Processing

![VQA examples](https://visualqa.org/static/img/vqa_examples.jpg)
## Motivation
We have decided this project because we considered that being able to answer a question from an image using AI it's 'cool' and, more importantly, it is a project that due to the multimodal approach requires that you must understand two of the most important disciplines in AI-DL: vision and language processing.

In addition it's an area relatively new.  (2014 - Papers 2015) with plenty of opportunities for improvement and several possible business applications: 
* Image retrieval - Product search in digital catalogs (e.g: Amazon)
* Human-computer interaction (e.g.: Ask to a camera the weather)
* Intelligence Analysis
* support visually impaired individuals

## Proposal

There are several possibilities to address VQA and this project has been based on VQA 2015 paper: https://arxiv.org/pdf/1612.00837.pdf

Use the paper model as a base to introduce variations in the composing elements:

![](images/model-puzle.png)

- using a different (newer) model for vision
- using a different strategy for the language channel
- other variations (ie. replace tanh by relu as non linearity)

Implement base and tuned models and check their metrics
Choose final model and analyze results

## Milestones
- Base model
- Tuned models
- Final model

# Working environment
One of our earliests decissions was to discard Tensor Flow in favour of **Pytorch**. We have had very few labs with Tensor Flow at that time and the learning curve for Pytorch seemed less steep.
**Google Colab** became quickly our preferred environment due to its easy of use, allowing us to combine text,code and graphics in a single document. The possibility of using GPUs became quickly a must as the size of the datasets got bigger.
In order to get shared access to our datasets, we stored them in a shared **Google Drive** folder wich could be conveniently connected to Google Colab backend using the `google.colab` module. After working this way for a while, we discovered that using OS access Colab possibilities (`!command`) to copy the dataset to the Colab's machine local disk was better in terms of performance.
# Models
Our models are based on the best performing one from paper [VQA: Visual Question Answering](https://arxiv.org/pdf/1505.00468.pdf):
![](images/paper-model.png)

The model had two branches: one for vision (image) and one for language (question). For the language branch, several alternatives were mentioned, all of them involved training the question embeddings. At least two different ways of combining the information from both channels were also tested.
With all this variants, a more general approach to the model can be seen as a framework with interchangeable pieces.

## Initial models
The best performing model from the paper used a pretrained  **vgg-16** for the vision piece and a 2 layer LSTM for the language channel. As we haven't gone through the NLP part of course at that time and, in order to have a complete working model as soon as possible, we decided to keep the original vision piece but go for pretrained embeddings for the whole question. Looking for a suitable model and after discarding the word oriented alternatives (Glove, Word2Vec), we found Google's [Universal Sentence Encoder](https://static.googleusercontent.com/media/research.google.com/ca//pubs/archive/46808.pdf). It provides 512 dim encodings for a sentence(question) and there is a Tensor Flow based [implementation](https://tfhub.dev/google/universal-sentence-encoder/4). After checking it worked fine from Google Colab and it did not collide with the rest of our Pytorch code, we used it to build our first model:

![](images/model-0100.png)

As the embeddings where of different lengths we decided to use concatenation to combine them (4096+512=4608). The final classifier was a funnel of three fully connected layers (4608-1024-512-20).

To train this initial model, and due to computing resource limitations, we created a custom dataset. It was a very reduced version of the original COCO dataset. It was actually a random selection of 1000 triplets (image, question, annotation) including only answers of type `'number'`and values from 1 to 20 extracted from the [2014 validation dataset](https://visualqa.org/download.html).
With this first dataset we wanted to have a fixed number of outputs at the classifier (20) and a manageable dataset size. We used 750 samples for training and 250 for validation. We developed a specific `Dataset`object to combine image, question text and annotation extracted from the COCO dataset files.

We managed to train the model but showed signs of overfit:

![](images/model-0100-metrics.png)

We tried to improve the metrics by adding `dropout` layers between the fully connected layers of the classifier and batch normalization but 
the accuracy peaked around 25% and didn´t get better, even if the model was trained for more epochs.
A quick analysis of the dataset showed it was highly inbalanced, with answers 1, 2 and 3 outnumbering the others alltogether. We used a weighted loss using #annotations/freq(annotation) to see what was the impact on the accuracy. The result was: more classes participated in the accuracy but the overall accuracy didn't improve.

A new dataset of 7500 samples (+ 250 for validation) with same selection as before (answers 1-20). Accuracy raised to 35%. Below expectations considering the task had been simplificated.

Using a similar dataset but including only yes/no answer type:

![](images/model-0100-metrics2.png)

Accuracy peaked close to 65% which is also belo expectations as it is only 15 points over random answer.

| Metric | Value |
| ----------- | ----------- |
| Throughput | 24.8 samples/s |
| Epochs | 20 |
| Accuracy (train) | 24.8% |
| Accuracy (validation) | 79.8% |

This model's code can be found [here](model-colabs/Model100.ipynb).

## Model variations
- Resnet for vision
- Concat vs pointwise
- Other variations
## Tuning the vision channel
### Model baseline with Resnet

As part of the intial research on the Vision channel we started considering the alternatives that we could have to VGG:
![](images/VisionAlternatives.png)

from all the alternatives we selected to focus on RESNET as it is one of the most common networks used today and with less computation efforts delivers better accuracy than VGGs

As part of this evaluation we wanted to check also the relevance or not of the size of the RESNET output as we were having the intuition that larger output should imply more features/resolution to help the rest of the network to deliver better accuracy.
Additionally based on previous results we were also getting the intuition that the network was not delivering better accuracy as the Dataset was too small and for this reason we created a new larger dataset

So the experiment was looking to understand:
1. What is the baseline performance for the model when RESNET it's being used? 
2. Does it matter the output resolution of vision channel?
3. Increasing the Dataset would the network perform better and leverage more powerful architectures?

Based on this we have selected the following: 
|RESNET| Output Features |
|------|----------------
|RESNET18|512|
|RESNET50|2048|

Experiment Results:

| Model Name | Model Architecture | Dataset A | Dataset C | Dataset D |
|---- |--- |:----:|:----:|:----:|
|1.e | RESNET18 (512), concat 1024, 1024->4096->1024->n Classes Batchnorm, dropout|34%|33.4%|34%|
|1.f | RESNET50 (2048), concat 2560, 2560->4096->1024->n Classes Batchnorm, dropout|33.6%|33.68|36.4%|

![](images/resnet18vsresnet50.png)

Looking at the results obtained and the ones that we had when using first models based on VGG we got the intuition that the model that was using RESNET was getting better accuracy speacilly when the dataset was bigger however at this point of the research we coudln't yet confirm it as the models were having other differences. This is going to become a new hypothesis to be validated with additional tests.

## Is the model actually learning something?
At some point of the research we were having doubts about if the model was actually learning as much as it could learn although the losses were indicating that it was learning. The main reason for this concern it's also because in just a few epoch the model stopped increasing accuracy in validation.

We came back to check what where the results of the models participating in VQA challenges. 
VQA organization is doing challenges every year and you could find the results on the 'Challenge' option of the menu and then click on 'Leaderboards'. For instace the results for 2016 are in You could find in https://visualqa.org/challenge_2016.html. We have been using these results as our model was starting from the architecture proposed on their paper from 2015.

![](images/VQAOpenEndedChallengeLeaderboard2016.png)

Comparing our 36.4% with Dataset D and the results of this table and bearing in mind our hypothesis that a larger dataset would help us to increase we got the intuition that we were going in the right direction and we could accept that the model was learning properly.

In order to perform an extra validation we decided then to focus on the 'Yes/No' type of question as we were doing the assumption that creating a dataset only focusing on answering (Yes/No) will be acting as a larger dataset as we'll have on this case thousands (10K) of images but just for 2 classes.


So the new experiment consisted on creating a new 10K dataset with questions where the answer only could be "Yes/No".
For this experiment we used the model 1.f introduced before.
See below the results in a similar format than the one used for VQA organization.

|Model|Dataset 10K Number (1-20) | Dataset 10K Yes / No | Average|
|---|:---:|:---:|:---:|
|1.f|36.4%|68%|52.2%|

Based on these results we got confidence with our model and make us focus about how we could scale to train with larger datasets.

## How important is to train only classifier vs train also vision?
As we were looking for options to scale the training but using the same infrastructure we decided to investigate if training also the vison channel was delivering better results. Overall the idea was that if it is not bring better results to train together classifier and vision channel then we could process the images embeddings before the training so the performance and resource consumption will be lower during training.

For this experiment we have used again our model 1.f and build the variants 1.fB and 1.fC as described below:

|Model|RESNET|
|---|---|
|1.f| Resnet50 train|
|1.fB| Resnet50 frozen|
|1.fC| Resnet50 last Convolutional train|

We perfom the evaluation of these 3 models for the datasets D & E and see below the results:

![](images/ResnetTrainvsFrozen.png)

These results seem to confirm that training the vision channel with the classifier better accuracy results in validation can be achieved.

## Classifier Architecture
At the initial stages of the projects we started with a simple classifier that was taking as input the information from Vision and Langugage channels and reducing it directly to the number of classes that we had to predict for that dataset.
After a few tests we got the intuition than the size of the layers and also the number of layers in addition to techniques like dropout and batchnorm could help us to increase.

For this reason we run an experiment to compare 2 models where the unique difference was the architecture of the classifier.
The objective was to compare if the accuracy of the model increases adding one intermediate level on the classifier smaller than the input size and large than the output in order to smoothly decrease from the 4096 to the output (20)

For this experiment we have used our models 1.b & 1.c and Datasets A and C

| Model Name | Model Architecture |
|---- |--- |
|1.b | RESNET18 (512), pointwise 512, 512->4096->n Classes|
|1.c | RESNET18 (512), pointwise 512, 512->4096->1024->n Classes|

![](images/AddFClayer.png)

As we can see this multilayer approach with progressive reduction of the number of features in the the input to the output it's increasing the accuracy of the model.

## Splitting the model
We realised a bigger dataset would be the best cure for our model's overfit and might bump up the metrics but the training was getting considerably long (ie. 100 minutes for 7,500 samples and 30 epochs) and after many long trainings we were sometimes banned to use Google Colab with GPU for some hours. In a Computer Vision lab we learned the trick of precalculating the image embeddings once and reuse them during the training process.
To implement it, we splitted the model in 2:
![](images/model-split.png)

First half uses our custom dataloader but instead of feeding the model, it stores the precalculated embeddings in lists (image and question). These two lists of tensors are then stored to disk using `torch.save`. Along with these two lists, a list with the annotations, a list index-to-annotation and a dictionary with additional information about the sample are also stored.
Second half uses a `TensorDataset`to load the precalculated embeddings after retrieving the lists with `torch.load`and feeds them to the rest of the model (combination + classifier).
This change really bumped up the overall performance increasing the throughput from 50 samples/sec to 5,000 samples/sec. Additionally we've also been able to use batch sizes as big as 400 while before we were restricted to a maximum of 30. This improved performance allowed us to move from 10k to 100k datasets.

On the down size, precalculating the image embeddings prevents from finetuning the vision model (include it in the training (all or part of it,usually the final layers) so it adapts to our images.

## Training with 100k dataset
- Results
## Tweaking the language channel
- lstm
- glove+lstm
# Result analysis
- Accuracies by question type

| Question type |  # questions  |  Hits  | % T1 |  Hits top 5  | % T5 |	
| --------- |  ---------:  |  ---------:  | :---------: |  ---------:  | :---------: |
| what sport is             | 121 | 101 | 83,5% | 116 | 95,9% | 
| is there a                | 522 | 431 | 82,6% | 521 | 99,8% | 
| what room is              | 100 | 81 | 81,0% | 96 | 96,0% | 
| can you                   | 97 | 67 | 69,1% | 95 | 97,9% | 
| is the woman              | 129 | 89 | 69,0% | 127 | 98,4% | 
| is there                  | 350 | 240 | 68,6% | 347 | 99,1% | 
| is the person             | 82 | 55 | 67,1% | 78 | 95,1% | 
| do                        | 182 | 122 | 67,0% | 180 | 98,9% | 
| does the                  | 357 | 238 | 66,7% | 347 | 97,2% | 
| is it                     | 410 | 268 | 65,4% | 405 | 98,8% | 
| is that a                 | 84 | 52 | 61,9% | 83 | 98,8% | 
| what animal is            | 102 | 62 | 60,8% | 80 | 78,4% | 
| is the                    | 1969 | 1185 | 60,2% | 1900 | 96,5% | 

| Question type |  # questions  |  Hits  | % T1 |  Hits top 5  | % T5 |	
| --------- |  ---------:  |  ---------:  | :---------: |  ---------:  | :---------: |
| what does the             | 226 | 41 | 18,1% | 58 | 25,7% | 
| who is                    | 135 | 23 | 17,0% | 65 | 48,1% | 
| where are the             | 137 | 23 | 16,8% | 48 | 35,0% | 
| where is the              | 452 | 73 | 16,2% | 165 | 36,5% | 
| what time                 | 186 | 30 | 16,1% | 61 | 32,8% | 
| how                       | 284 | 38 | 13,4% | 89 | 31,3% | 
| why                       | 151 | 15 | 9,9% | 35 | 23,2% | 
| why is the                | 76 | 6 | 7,9% | 12 | 15,8% | 
| what is the name          | 79 | 5 | 6,3% | 7 | 8,9% | 
| what number is            | 74 | 3 | 4,1% | 8 | 10,8% | 


- Interesting data
- Interesting samples
-- Elephant butt
-- Ocluded individuals in how-many questions
- Interpretation
# Learnings
- Multimodal has helps us:
    - Consolidate knowledge around Vision
    - Consolidate knowledge around NLP
    - Learn different possibilities about how to combine the results of different networks (e.g.: pointwise, concat, Before a dot product, normalize the vectors)
- Dataset:
  - - Dataset size is critical - The bigger the better. Balance, Optimization based on pre-calculated embeddings
- Classifier:
  - Augmentation Layer, Multilayer FC feature reduction (funnel), Dropout, Batchnorm
- Vision:
  - VGG, Resnet
- Language:
  -Google sentence encoder, Glove, Own encoders
- Train:
  - Loss, train accuracy and validation accuracy plots interpretation
  - How to combine Batch size, lr and lr schedulers
  - Transfer learning at different levels (e.g.: frozen pre-trained network, partial training or a pre-trained network, full training of a pretrained network)
  - Split the model if image embeddings are static to reduce training times and resources needed
- Visualitzation:
  - Plots, Tensoboards
- General:
  - Learn and progress by defining hypothesis, run experiments and extract conclusions and new hypothesis.


# Next steps
- Use image embedding as initial context for the LSTM 
- Add attention to the language branch
- Add attention to the language branch but using the image embedding
- Enrich image information with object detection/object segmentation info
- Train Vision in addition to classifier
