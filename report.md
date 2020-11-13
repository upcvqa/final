test branch Jordi
# Introduction
![VQA examples](https://visualqa.org/static/img/vqa_examples.jpg)
## Motivation
## Proposal
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
- Resnet partially trained

As part of the intial research on the Vision channel we started considering the alternatives that we could have to VGG:
![](images/VisionAlternatives.png)

from all the alternatives we selected to focus on RESNET as it is one of the most common networks used today and with less computation efforts delivers better accuracy than VGGs

As part of this evaluation we wanted to check also the relevance or not of the size of the RESNET output as we were having the intuition that larger output should imply more features/resolution to help the rest of the network to deliver better accuracy.
So the experiment was looking to check:
1. Is it better RESNET than VGG?
2. Doe it matter the output resolution of vision channel?


RESNET18

RESNET50






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

- Interesting data
- Interesting samples
-- Elephant butt
-- Ocluded individuals in how-many questions
- Interpretation
# Learnings
- Before a dot product, normalize the vectors
- Split the model if image embeddings are static to reduce training times and resources needed
- Dataset size is critical. The bigger the better.
# Next steps
- Use image embedding as initial context for the LSTM 
- Add attention to the language branch
- Add attention to the language branch but using the image embedding
- Enrich image information with object detection/object segmentation info
