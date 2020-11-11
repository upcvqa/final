# Working environment
One of our earliests decissions was to discard Tensor Flow in favour of **Pytorch**. We have had very few labs with Tensor Flow at that time and the learning curve for Pytorch seemed less steep.
**Google Colab** became quickly our preferred environment due to its easy of use, allowing us to combine text,code and graphics in a single document. The possibility of using GPUs became quickly a must as the size of the datasets got bigger.
In order to get shared access to our datasets, we stored them in a shared **Google Drive** folder wich could be conveniently connected to Google Colab backend using the `google.colab` module. After working this way for a while, we discovered that using OS access Colab possibilities (`!command`) to copy the dataset to the Colab's machine local disk was better in terms of performance.
# Models
Our models are based on the best performing one from paper [VQA: Visual Question Answering](https://arxiv.org/pdf/1505.00468.pdf):
![](images/paper-model.png)

The model had two branches: one for vision (image) and one for language (question). For the language branch, several alternatives were mentioned, all of them involved training the question embeddings. At least two different ways of combining the information from both channels were also tested.
With all this variants, a more general approach to the model can be seen as a framework with interchangeable pieces.

![](images/model-puzle.png)
## Initial models
The best performing model from the paper used a pretrained  **vgg-16** for the vision piece and a 2 layer LSTM for the language channel. As we haven't gone through the NLP part of course at that time and, in order to have a complete working model as soon as possible, we decided to keep the original vision piece but go for pretrained embeddings for the whole question. Looking for a suitable model and after discarding the word oriented alternatives (Glove, Word2Vec), we found Google's [Universal Sentence Encoder](https://static.googleusercontent.com/media/research.google.com/ca//pubs/archive/46808.pdf). It provides 512 dim encodings for each question and there is a Tensor Flow based [implementation](https://tfhub.dev/google/universal-sentence-encoder/4). After checking it worked fine from Google Colab and it did not collide with the rest of our Pytorch code, we used it to build our first model:

![](images/model-0100.png)

As the embeddings where of different lengths we decided to use concatenation to combine them (4096+512=4608).To train this initial model, and due to computing resource limitations, we created a custom dataset. It was a very reduced version of the original COCO dataset. It was actually a random selection of 1000 triplets (image, question, annotation) including only answers of type `'number'`and values from 1 to 20 extracted from the [2014 validation dataset](https://visualqa.org/download.html).
With this first dataset we wanted to have a fixed number of outputs at the classifier (20) and a manageable dataset size. We used 750 samples for training and 250 for validation.
A quick analysis of the dataset showed it was highly unbalanced, with answers 1, 2 and 3 outnumbering the others alltogether.

We managed to train the model but showed signs of overfit:

![](images/model-0100-metrics.png)

We tried to improve the metrics by adding `dropout` layers between the fully connected layers of the classifier and batch normalization but 
the accuracy peaked around 25% and didn´t get better, even if the model was trained for more epochs.

| Metric | Value |
| ----------- | ----------- |
| Throughput | 24.8 samples/s |
| Epochs | 20 |
| Accuracy (train) | 24.8% |
| Accuracy (validation) | 79.8% |

## Tuning the vision channel
## Splitting the model
## Tuning the language channel
