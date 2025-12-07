# From Speed to Stability: How SLMs can be adapted to deliver efficient, robust and instruction-aligned classification in a single forward pass

## TLDR

## Motivations

When training a multimodal reranker, I modeled it as a binary classification problem with a transformer backbone. 

Because of the multimodal aspect and the very poor performance of encoder only models compared to decoder only (even for embeddings, see Colpali) in mutlimodal embeddings and representations - mainly because of the discreptencies between training of both type of models and because of the real multimodality in the input being treated almost only in VLMs - I was "forced" to use Causal model for this classification task.

However, an interesting point I realized is that training only the core representation model - so only the linear layers of attention and FFN and not the embedding and LM head layer - performed the best on test sets and benchmarks after training (meaning better performance and less overfitting). Also using the LM head (and extracting logits - but I will focus on that later) worked better than plugging and training a specific MLP as an output to the last token embedding.

This can be explained because instead of tuning the last layer on a specific dataset and forcing it to fit the - most of the time - small number of example. By using the pre trained LM head for logits yes and no present several advantages :
- We can leverage the instruction following and in context learning capabilities of language model by explaining the classification task and expressing the meaning of the classes. Instead of letting the model understand it through backpropagation and "waste" some compute and training example to make it figure out the task and then improve at the task, we can directly start at the becoming good at the task.
- Since we are training and modifying only the core model and not the first layer (embedding layer) and last layer (frozen lm head instead of fully trained MLP, now most of the time tied to the embedding - see section of this blog on tie embeddings) the train model is more robust to overfiting and better in OOD examples of the same task (+ LoRa already reduces overfitting in the core model). We can see it as : we are not modifying the "decision" but only the "represention" (like intermediate vectors / embedding) that are used for the "decision" (the final classification).

This was true for the binary classification task and multimodal - which may be a niche in the industry. But is it true for more complete task (multiclass classification) in the text only area, where embeddings and encoder only models are really good and mainly used in classification tasks (intent classification, NER, ...) plugged to an MLP ?

I decided to run several ablations on this subjects to compare methods for multiclass text classification to the single forward pass architecture I used for the reranker.

## *Classifications strategies*

I will first describe the different techniques that are mainly used when you want to perform text classification (binary or multiclass).

### Embedding + MLP

The main technique used for classification tasks (intent classification, NLU, NER, emotion classification, ...) is to plug a text embedder to a decision maker, so most of the time, a transformer encoder represent the text input into a rich vector and then an MLP extract the classes with a final softmax or sigmoid activation depending on the task. It was mainly done with biderectional encoders (derivated from BERT) that are pretrained on Mask Language Modeling and postrained on sentence representations through pooling and constrative learning (like multilingual-e5 or gte-embedding).

![alt text](bi_embedding_mlp_classifier.png)

After that, encoder only models where also given the "instruction" capabilities, meaning you can explain deeper your task to the model so the embeddings are a bit more adapted to what you want to do (for example you could explain the classification task or the retrieval task and say that you give the query or the document, see Jina's embedding v4 where they even train a LoRa per task ADD LINK or multilingual-e5-instruct).

Nowadays, embedding can also be decoder / causal embeddings when SLMs are postrained following constrative learning recipe. This is the case of qwen3-embedding-0.6B which transforms qwen3-0.6B into an embedder - leveraging instruction - by applying the gte embedding constrative learning recipe. Those are the best embedders in most of the tasks according to the MTEB leaderboard (ADD LINK) - this is actually the closest technique to what I present in this blog except you need a fully trained MLP.

![alt text](causal_embedding_mlp_classifier.png)

### Semantic similarity

Most of the embedders / encoders are trained for similarity tasks (because of the focus on retrieval task in benchmark, use cases and training). So you can leverage those model as bi-encoders where you encode the input and the description of all classes separatly and create a score using cosine distance. You can precompute the classes vectors at inference time and apply a softmax to the scores to get a probability.

![alt text](embedding_similarity_classifier.png)

### Reranker / Natural language inference pairs

Another technique used in NLP is called Natural Language Inference and is really close to reranking tasks (see for example the paper : Political DEBATE: Efficient Zero-shot and Few-shot Classifiers for Political Text). The concept of NLI is almost as the Next Sentence Prediction of the original training of BERT. You present a text and a description of class and the model classifies the relevancy between the 2 subjects. So you can use text reranker in this setup, such as qwen reranker.

![alt text](reranker_pairs_classifier.png)

The only issue is that on the opposite of all the previous techniques where you need a single forward pass to classify at inference time, NLI and rerankers work as cross-encoders meaning you cannot precompute anything. You need to perform each pairs of input / class for every input. In multiclassification problems, it can become very expensive (you multiply your compute budget by the number of classes).

Even if cross encoding is robust (that's why rerankers are use as a final layer before generation in RAGs), it makes it unrealistic for fast classification tasks - even more on CPUs.

### Naive LLM classification + parsing

Now, we will start speaking about LLMs. When people want to classify using LLMs, I used to become irritated. Why would you use big models to just get a single class which is a sinple score. You need to let the model do all its generation / autoregressive process, beg him to return your class name in between some irrelevant tokens, try some regex or fuzzy match to find your class tokenS inside the completion and you can barely have a real probability of it.

### SLM last token hidden state + MLP

A more clever way to do this task is to use a single forward pass to get the last token hidden state (just like it is done in qwen embeddings) and plug it to a fully trained MLP. There, you can fine tune the model and it is not an autoregressive task.

![alt text](slm_mlp_classifier.png)

### SLM single forward pass

Finally what I saw worked the best in the binary classification problem + multimodal task was to use a Small Language Model (the size of the encoder only), perform a single forward pass and extract the logits of the class tokens from the LM head. There, you can instruct the model with the task and ask him (+ train him) to answer directly with the class token. If you do a softmax with only the X logits representing your X classes (and present in the prompt), you have access to real probabilities of the classes.

One constraint is that you need to find a way to present your class with DISTINCT first logit. 

A variant is to use proxy to your classes by giving correspondances (to letters A,B,C,D, ...) in the instruction and use those proxy logits as classes.

![alt text](slm_sliced_head_classifier.png)

With that, you can leverage strong training of Causal models (multilingual, knowledge, huge pretraining, post training and RL on task), intruction following abilities (you can skip the adaptation to the task because you explain the task in the prompt and go directly to model improvment) and it costs a single forward pass and provides you real probability of classes.

## *Comparison, Training Setup and Evaluation Setup*

### Datasets
To evaluate those assumptions and compare strategies in a relevant way, I used 2 datasets :
- A first German dataset for 

### Models
MTEB classification screen

### Hyperparameters, LoRa and prompts
Datasets
Models
Git

Allemand + 8 classes

Anglais + 6 classes

## *Results*

### Global evaluation on german task
Parler de moins d'overfit

### Focus on English task

### SSLMs future
Baguetotron, Monad, ...

## Focus on tie embeddings
Layers, Memory and Prompts
Other advantages (cite Gemma paper)

## Optimization and cost
Slicing
FLOPS
GPU RAM Bandwith