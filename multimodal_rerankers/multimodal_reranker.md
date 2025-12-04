# Building and evaluating Multimodal Rerankers

## TLDR

I trained a multimodal reranker based on Qwen 3 VL 2B that outperforms Jina Reranker M0, optimized both its inference speed and model size, built a collection of datasets to evaluate multimodal rerankers, and experimented with applying RL to reranking (promising and worth trying, but still inconclusive).

* Code : https://github.com/UlrickBL/multimodal_reranker#
* Models : https://huggingface.co/collections/UlrickBL/multimodal-reranker
* Benchmarks : https://huggingface.co/collections/UlrickBL/vidore-benchmark-reranker-adapted
  
## **Multimodality in retrieval and the need for rerankers** 

RAGs have evolved in several directions over the last few years. From simple “retrieve and generate,” the field has moved toward agentic RAG, deep research, and multimodality. When retrieving images, the most common approach is to use a query/document embedder and a vector database. It works the same way for multimodality, except documents are no longer text chunks but page images.

Colpali: https://arxiv.org/abs/2407.01449

In text-based RAGs or deep research systems, a key component is often a reranker. Unlike a bi-encoder, which encodes the query and document in two separate forward passes, a reranker “cross-encodes” the query and document(s) together in a single forward pass. This is more expensive (because you cannot pre-encode documents without knowing the query), but much more powerful since the model directly scores the relevance of each query–document pair.

Because of the higher cost, rerankers are used on far fewer documents than the retriever (you retrieve, then rerank, then generate). For example, you might retrieve the top 25 documents with a bi-encoder, rerank those 25, and then select the top 5 or even top 1 for generation. This reduces the context size (which makes it cheaper and more efficient) and improves relevance.

Multimodal rerankers are not commonly trained or used yet, but they can be even more useful than text-only ones, as VLMs tend to struggle when presented with multiple images at once (due to layout, multidimensionality, and bidirectionality issues). 

Jina even performs deep search with a reranker, stating: “Semantic Relevance: For question-based searches, we use Jina Reranker to assess the semantic match between the question and the textual info of each URL, which is a classic reranking problem.”
https://jina.ai/news/a-practical-guide-to-implementing-deepsearch-deepresearch

Multimodal RAGs are being promoted as the future of RAG by many companies because parsing, chunking, layout detection, and OCR are heavy and often low-quality. Many corporate use cases are visually rich (plots, tables, images, slides, diagrams, etc.) and are almost impossible to fully represent as text. See Lighton’s blog “RAG is dead, long live RAG”:
https://www.lighton.ai/lighton-blogs/rag-is-dead-long-live-rag-retrieval-in-the-age-of-agents
,
Cohere: https://cohere.com/blog/multimodal-embeddings
,
Jina: https://jina.ai/news/jina-embeddings-v4-universal-embeddings-for-multimodal-multilingual-retrieval/
.

Additionally, DeepSeek’s OCR work showed that images can even achieve better compression than text.

While multimodal retrievers like Vidore and Colpali are becoming common, very few multimodal rerankers exist. The most well-known one is Jina Reranker M0 (https://jina.ai/news/jina-reranker-m0-multilingual-multimodal-document-reranker/
), which is not fully open source.

Moreover, there is still no solid benchmark or leaderboard that allows the few teams working on this topic to compare and evaluate their models.

## **Text reranker architecture**:

I first discovered how rerankers work effectively through Alibaba’s M-GTE paper: https://arxiv.org/html/2407.19669v1

The base model is presented as the same for both the M-GTE retriever (a bi-encoder) and the reranker. However, the reranker is post-trained to output a scalar that represents the relevance between a query and a document in a cross-encoder manner. This is similar to how BERT was trained on the Next Sentence Prediction (NSP) task, where it was given sentence pairs and asked to score whether they follow each other, and also similar to the Natural Language Inference (NLI) task.

![image](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/voKr-w13uZjnCa8ASZ4WV.png)

The goal is not to generate a vector embedding but to produce a single scalar: a probability of relevance.

In the multimodal space, even retrievers are often built on fully autoregressive or decoder-style architectures with some additional tweaks, instead of using a bidirectional encoder like in text models (even if this is starting to change with models such as Qwen Embedding for text). For example, the original Colpali combines a SigLIP multimodal encoder with PaliGemma, where the embedding is created from the hidden states of tokens and patches. This is needed because the model must represent language and cross-modality together, and pretrained VLMs already provide good grounding capabilities such as cross-modal alignment and VQA, even if they lack full bidirectionality.

The first multimodal reranker I encountered, and still the SOTA to my knowledge, is Jina reranker m0 and is based on Qwen2-VL 2B. It uses an MLP layer that outputs the scalar relevance score from the hidden state of the final token in the context, which contains the concatenated query and image document.

![image-1](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/4eXSCD48cDzLVJk0y_YbK.png)

However, when Qwen released their first reranker based on a decoder backbone, they introduced a clever way to score relevance by using the probability of the "yes" logit when the model is prompted with a reranking instruction.

![image-2](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/VBjTj4p2wcSC7SuhNqAj1.png)

This makes training lighter because the classification layer does not need to be trained from scratch, since the model can simply use its existing LLM head.

## **Benchmark and metrics**

Before talking about training, let’s speak about benchmarks, metrics and cost, in order to enable cost-efficient, reproducible and realistic evaluations.

I created a collection of datasets for a reranking benchmark based on the VIDORE v2 benchmark and IBM Real MM RAG, adapted to evaluate reranker models in a multimodal retrieval setting. The dataset includes a corpus of image data, a set of natural language queries, and the top 25 retrieved images returned by a mid-performance multimodal retriever. This setup simulates a realistic retrieval environment where the reranker must learn to surface relevant items that may not already be ranked highly.

The purpose of this benchmark is to:

* Evaluate rerankers independently of the retriever by fixing the retriever outputs.

* Focus on the ability of rerankers to identify relevant samples from mid-quality retrieval sets.

* Provide detailed statistics on retrieval and relevance structure to better understand model behavior.

* Offer a challenging but meaningful setting for reranking by using a retriever with known mid-level performance on the VIDORE v1 leaderboard.

The retriever used is Alibaba-NLP/gme-Qwen2-VL-2B-Instruct (ranked around top 23 with 87.8 accuracy at the time).

The retriever was used to embed the full image corpus of each dataset (each sub-corpus processed separately). For each query, the retriever computed similarity scores and returned the top 25 most similar images. These 25 candidates were labeled using the ground truth relevance annotations from VIDORE v2 or Real MM RAG. Only retrieved items are considered during evaluation. Relevant items that the retriever failed to retrieve are ignored, so the evaluation focuses specifically on reranking.

For Real MM RAG, I used the rephrased level 1 version to increase difficulty and relevance. This means that queries usually do not contain words that appear in the image, making the task a genuinely visual or semantic reranking problem instead of a keyword match.

Here is the list of datasets with some statistics:
| Metric / Value                                   | vidore_benchmark_2_esg_reports_human_labeled_v2_reranker_adapted | vidore_benchmark_economics_reports_v2_reranker_adapted | vidore_benchmark_arxivqa_reranker_adapted | REAL-MM-RAG_FinSlides_BEIR_reranker_adapted |
|--------------------------------------------------|------------------------------------------------------------------|---------------------------------------------------------|--------------------------------------------|-----------------------------------------------|
| Number of queries                                | 52                                                               | 232                                                     | 500                                        | 50                                            |
| Corpus size                                      | 1540                                                             | 452                                                     | 452                                        | 2280                                          |
| Average # relevant images per query              | 2.46                                                             | 15.64                                                   | 1.0                                        | 1.0                                           |
| Average # retrieved relevant images in top 25    | 1.73                                                             | 5.77                                                    |                                            | 0.96                                          |
| % of queries with at least one relevant retrieved| 91.25%                                                           | 93.97%                                                  |                                            | 96.0%                                         |
| Avg. position of first relevant image            | 3.53                                                             | 3.03                                                    |                                            | 3.52                                          |
| Avg. position of last relevant image             | 6.82                                                             | 16.78                                                   |                                            | 3.52                                          |
| NGCD@5 (Normalized Gain Cumulative Discounted at 5) | 0.6424                                                          | 0.4951                                                  |                                            |                                               |


Collection: https://huggingface.co/collections/UlrickBL/vidore-benchmark-reranker-adapted

To evaluate ranking systems, two metrics are mostly used: NDCG and MRR. While they both measure how high up the list the good stuff is, they handle penalties and relevance very differently.

### MRR (Mean Reciprocal Rank)

MRR is the speed metric. It cares about one thing: how fast did the user find the first relevant result.

It is strictly binary (an item is either relevant or not) and it stops counting the moment it finds a match.

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$$

Where $rank_i$ is the position of the *first* relevant item for query i.

### NDCG (Normalized Discounted Cumulative Gain)

NDCG is the quality metric. It looks at the entire list (up to a cutoff k) and rewards the model for placing high-relevance items at the top.

Unlike MRR, NDCG handles graded relevance (perfect match vs partial match) and accounts for multiple positives (which is the case of some benchmark of the collection).

NDCG is calculated by taking the Discounted Cumulative Gain (DCG) and dividing it by the Ideal DCG (IDCG), which is the score of a theoretically perfect ranking.

$$DCG@k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)}$$

$$NDCG@k = \frac{DCG@k}{IDCG@k}$$

To truly understand the difference, let's look at a scenario where there is only one relevant document in the whole list (in Real MM Rag for example).

* MRR decays linearly (1/r).
* NDCG decays logarithmically (1/log_2(1+r)).

Here is how the scores drop as the relevant item slips down the ranking:

| Rank ($r$) | NDCG (Single Positive) | MRR (Single Positive) | Analysis |
|---:|---:|---:|:---|
| **1** | 1.000 | 1.000 | Both metrics give a perfect score. |
| **2** | 0.631 | 0.500 | MRR slashes the score in half immediately. |
| **3** | 0.500 | 0.333 | |
| **4** | 0.431 | 0.250 | |
| **5** | 0.387 | 0.200 | |
| **10** | 0.289 | 0.100 | |


While the table above shows a single positive, real-world retrieval benchmarks often contain multiple positive documents for a single query.

In these cases, MRR is insufficient because it ignores everything after the first hit. If a system retrieves 5 relevant items, but puts them in positions 1, 50, 51, 52, and 53, MRR gives it a perfect score (1.0) because the first hit was at rank 1.

NDCG, however, rewards the model for packing all relevant items as high as possible. Since our goal in this project is to build a robust reranker that surfaces all relevant context, **NDCG@5** will be our primary metric.

## **Training strategy and experiments :**

My training had several goals and limitation :
- Use very few data to experiment a lot and show that we can increase performance quickly and easily
- Use very few compute so the training and adaptation and cost efficient
- Compare MLP based approach (the Jina one) with Logit based approach (the Qwen one on text reranker)
- Optimize as much as possible inference and memory

### Model 1 : Qwen 2.5 VL 3B and in batch negative


For the first version of the model, I used Qwen 2.5 VL 3B as the base model and I randomly subsampled 2000 rows from the Vidore training dataset used in the original Colpali. The dataset contains pairs of queries and images: UlrickBL/vidore-subset-train-2000.

To enable efficient training and maximize batch size on a single GPU, I used LoRA with rank 16 and alpha 32 on all linear layers of the attention blocks (Q, K, and V) and on the FFN layers (up projection and down projection). This provides good training capacity and adaptation to the task. I did not train the embedding layer or the final projection layer since I wanted to use the pretrained LM head that produces the logits.

The training hyperparameters that worked best for this setup were a batch size of 2 with in-batch negative mining, using 1 negative per sample. This means we have 2 pairs, and by permuting the pairs we create 1 negative example per sample, resulting in an effective batch size of 4.

For the rest of the hyperparameters, the optimizer was AdamW with a learning rate of 5e-5, a weight decay of 1e-2 and a max_grad_norm of 0.1, which showed a good learning curve.

I modeled the problem as predicting the probability associated with the "Yes" logit versus the "No" logit, using an instruction that asks the model to answer the question given the image and the query.

![image-3](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/0uYgCgghnbTUpcLDD0amH.png)

Then the hidden state of the last token is taken wr the attention mask for padding cases and the logit Yes and No are sampled from the output of the LM head.

The scalar output of the model is :
$$ \sigma(logit_{yes} - logit_{no}) $$

Since this is modeled as a binary classification problem, and because the model already produces a probability through the sigmoid, the loss used is simply binary cross entropy.

Training lasted for one epoch.

With this setup, I achieved quite good results compared to Jina Reranker m0 with a good inference speed :

| Dataset    | Jina Reranker m0 (Baseline) | QwenLogitReranker |
| ---------- | ------------------------ | ----------------- |
| UlrickBL/vidore_benchmark_economics_reports_v2_reranker_adapted  | 0.735                    | **0.799**         |
| UlrickBL/vidore_benchmark_2_biomedical_lectures_v2_reranker_adapted    | **0.763**   | 0.755             |
| UlrickBL/vidore_benchmark_2_esg_reports_human_labeled_v2_reranker_adapted  | **0.851**                | 0.820             |
| UlrickBL/vidore_benchmark_arxivqa_reranker_adapted | **0.767**                    | 0.747             |
| UlrickBL/vidore_benchmark_2_esg_reports_v2_reranker_adapted     | **0.920**   | **0.910**         |
| Inference time (4898*2810 image, T4 GPU)    | 2.212 s  | **1.161 s**         |

Despite smaller training data, limited diversity and reduced compute, the QwenLogitReranker shows competitive or superior performance, especially in Economics.

I also tried replacing the LM head with an MLP as Jina did, but the results were worse. This suggests that training this final layer from scratch requires more compute and more data, and that using the logits and the pretrained LM head is a good strategy for lighter training and efficient performance.

### A specificity of current VL models when training with HF transformers

I will diverge a bit to explain something important about the training process. Recent models such as Qwen 2.5 VL and Qwen 3 VL can handle several images inside the context and can also process several images inside the batch. The context is constructed in the following way:

The processor replaces each image in the context with a token called image_pad and stores the embeddings of all patched images from both the context and the batch inside a single array called pixel_values with shape (total_number_of_patches, embedding_dim). What allows the model to know when an image ends and how many images each batch element contains is the image_grid_twh field, which stores the grid shape for every image within each batch element.

![image-6](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/j4t80lyJwglOv3-1VSViI.png)

However, when training with the HF Trainer and going through the collator that prepares batches, mines negatives, and applies the processor, each element of the dictionary sent to the model must have the same number of rows. This reflects the batch size. To deal with this constraint, I had to apply a trick that consists in faking a batch of pixel values, padding them so that the result is a tensor, and then reshaping the proper pixel_values inside the model during the forward pass. One must also pay attention to tensor contiguity when doing this since I encountered issues when the tensor was not contiguous.

![image-7](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/XdTgDImF_JfLFPWCd0Z0C.png)

### Model 2 : Qwen 3 VL 2B and hard negative mining

To improve the model and attempt to outperform Jina, I switched to a more recent model that was released after my first experiment. It is smaller, which allowed me to increase the batch size.

I also combined the first dataset I used with in batch negatives and created a new dataset from the IBM REAL-MM-RAG_FinSlides_BEIR training set called UlrickBL/REAL-MM-RAG_FinSlides_BEIR_reranker_adapted, where I mined hard negatives using gme-Qwen2-VL.

This combination allowed me to use in batch negatives when sampling from the Vidore subset and hard negatives (the first negative example retrieved by gme-Qwen2) when sampling from the REAL-MM-RAG data. This provided the model with a mix of images and tables, including difficult and realistic cases, and it also allowed the model to learn reranking across different similarity levels, since in batch negatives are typically less similar than mined hard negatives.

I also made several changes to the hyperparameters. I used a batch size of 6 with 1 negative per example, resulting in an effective batch size of 12. I then added a gradient accumulation of 10, providing an effective total batch size of 120 for 2 epochs, with a learning rate of 1e-5 and a warmup of 20 steps.

![image-4](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/z93-GpGiy8SEpQVpaJNTx.png)

![image-5](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/2RnZWsyj-01wUD-wCaTxO.png)

Both training and eval were stable.

I evaluated the model on three hard and diverse benchmarks from Vidore V2 and on the rephrased version of the REAL MM RAG benchmark. The results show overall better performance than Jina Reranker M0.

| Dataset                  | Jina Reranker m0 | Qwen3RerankerMM-2B |
|--------------------------|------------------|---------------------|
| UlrickBL/vidore_benchmark_2_esg_reports_human_labeled_v2_reranker_adapted      | **0.851**            | 0.804               |
| UlrickBL/REAL-MM-RAG_FinSlides_BEIR_reranker_adapted (rephrased level 1)     | 0.873            | **0.906**               |
| UlrickBL/vidore_benchmark_economics_reports_v2_reranker_adapted   | 0.735            | **0.813**               |
| UlrickBL/vidore_benchmark_arxivqa_reranker_adapted                 | 0.767            | **0.778**               |


## **Optimizing inference** 

I will do another digression to speak about attention implementation difference between Qwen 2.5 VL and Qwen 3 VL in transformers. A major issue I had when doing the evaluation was trying to match the speed of Jina's model. I worked a bit on Kaggle T4 GPU and deactivated the flash attention and realized that the speed of Qwen 2.5 VL versus Qwen 3 VL, even in simple forward mode (not the generate mode) different by a factor of at least 10 which costed a lot to infer.

When FlashAttention is disabled, Qwen3-VL becomes significantly slower than Qwen2.5-VL.

This performance gap is not accidental it comes directly from the type of attention each model uses and the shape of the sequences they operate on.

When FlashAttention is not used (`_attn_implementation="eager"`), Qwen3-VL executes attention through a slow fallback mechanism designed to save memory, but at a massive compute cost.

It manually chunks the sequence using Python loops:

    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    splits = [
        torch.split(tensor, lengths.tolist(), dim=2)
        for tensor in (query_states, key_states, value_states)
    ]

    outputs = []
    # The bottleneck: A Python loop driving GPU operations
    for q, k, v in zip(*splits):
        out = eager_attention_forward(self, q, k, v)
        outputs.append(out)

    attn_output = torch.cat(outputs, dim=1)

This introduces three bottlenecks:
* Python loops kill GPU utilization. Instead of queuing one massive task, the CPU must interrupt to launch many small kernels sequentially.

* Each chunk equals one distinct attention call. This adds massive overhead compared to a single large matrix operation.

* Splitting q, k and v into many pieces and then stitching the results back together with torch.cat is an expensive memory operation on the GPU.

This chunked attention path is the dominant cause of slowdown.

In contrast, Qwen2.5-VL’s eager implementation is "vectorized." Even though it supports a massive sliding window of 32,768 tokens (which often covers the whole image + query anyway), it executes efficiently.

Qwen2.5-VL performs a single, vectorized GPU attention call, no manual splitting, no Python loops, no repeated kernel launches.

Yet, I used flash attention at the end.


### Slicing the vocab layer / LM head

Another tweak I found relevant to reduce the memory and the inference time is to slice the LM head. Indeed, the model only needs the logits of 2 tokens and not the other projections, it can reduce drasticly the size of the model, even compared to Jina MLP method without changing the performance.

Using the complete logit outputed by the LM head (since we don't sample token with softmax, temperature and so on) and slicing the 2 token that we care about like that :

![image-8](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/Mt6o5uH-JPEj_9sg3cscz.png)

Is the same as slicing the LM head when loading and use it like that in term of output :

![image-9](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/vFwdRMHtPKfL5VDKdQW1m.png)

However, in term of size of the model parameters are reduced by a lot.

Indeed, Qwen 3 VL has **2 127 532 032** parameters, the hidden dimension is 2048 and the vocab size is 151936, so the number of parameters in the LM head is 2048* 151936 = **311 164 928**. When slicing the LM head for the only 2 useful tokens, the size is of this last layer is 2048* 2 = **4 096**. The LM head was previously 14% of the model and now is negligable compare to the backbone and the size of the model is then **1.8B** instead of **2.1B**.

In Jina setting, the MLP has 2 layers, one with shape (hidden_dim,hidden_dim) and another that project the logit with shape (hidden_dim,1). The hidden dim of Qwen 2 VL 2B is 1536 and the same vocab size. So the LM was previously 1536 *151936  = **233 373 696** and is now 1536 *1536  + 1536 * 1 = **2 360 832** parameters.

Using the logits allows a better memory / FLOPs / size reduction in addition to have a pretrained layer instead of a fully initialized one. Be aware that it is not necessary a memory optimization since some models are using tie embedding meaning the embedding layer and the lm_head share the same memory (but it is still a reduction of "effective parameters" - related to number of operations, but not necessary "memory parameters").

In the end, with full optimization, for the same dataset, with a batch size of 25 pairs and 50 examples on a A100 40GB SXM4, UlrickBL/Qwen3RerankerMM-2B performs it in **1m41s** and Jina Reranker m0 in **2m12s** improving not negligably the inference time in addition to the overall performance and memory consumtion.

This methodology could be used for any classification task (like scalar reward modeling, intent classification, Judge, ...), I will speak about it in another blog.

## Reinforcement learning strategy and reranking environments

I tried a final strategy that did not work but could be interesting : Reinforcement learning for reranking.

The idea came by reading the paper of Jina with Last but not Late interraction : https://arxiv.org/abs/2509.25085

The idea is to apply RL with GRPO to the reranking task for visual documents.

When performing classic reranking, we need to roll out all pairs and backpropagate through each of them. In this setup, we mine in batch hard negatives to provide in batch information, but the loss is still computed at the pair level. This approach works for SFT or token loss objectives such as Binary Cross Entropy.

If we want to apply GRPO with an NDCG reward, we need to adapt existing libraries because we must compute X rollouts multiplied by Y pairs in order to obtain the NDCG for X. Then we must backpropagate through X multiplied by Y, since reranking is computed per pair of query and single retrieved document.

However, with Jina’s most recent method for text retrieval, everything fits inside one forward pass. This reduces the computation to X rollouts only. In this formulation, each pair contains the query along with positive and negative documents inside the same forward call. This makes the GRPO setup straightforward, and we can treat the task as a ranking problem where the model outputs a verifiable ordered list that can be evaluated with NDCG.

In practice, we provide QUERY + DOC1 + DOC2 + DOC3 and so on, and we prompt the model to output an ordered list of documents based on relevance, for example [DOC2, DOC3, DOC1]. We then compute NDCG on that ordering and derive the reward. Afterward, we perform X rollouts to compute the advantage.

I built an environment to achieve that on the prime intellect environment hub with the library verifiers : https://app.primeintellect.ai/dashboard/environments/ulrick-bl/reranker-vl

I tested different rewards but the final setup was with :

* reward_mrr : Compute MRR (Mean Reciprocal Rank) for predicted ranking.
* reward_parseable_list_only : Reward is 1 if output is exactly a parsable list and nothing else. Anything else (extra text, explanation, malformed) is 0.
* reward_valid_doc_tags :  Reward = 1 if list contains exactly the same DOC_X tags (no duplicates, no missing), and the count matches the number of images.

with weights 0.6, 0.2, 0.2

The model was prompted with this instruction :

    You are given a user query and several candidate documents (DOC_1, DOC_2, ...), each containing an image.
    Rank the documents from most to least relevant to the query.
    Return your ranked list exactly as a valid Python list like: [DOC_3, DOC_1, DOC_2, ...].
    Do not add explanations, text, or commentary.

I ran several experiments, but the loss and the diversity of the outputs were not sufficient. The rollouts tended to produce outputs that were too similar to one another, so the training signal remained weak.

![image](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/0D50ZFy0JnEF1gUNmslok.png)

My current understanding is that this happens because the number of presented documents is small. To keep inference efficient, the model is not encouraged to perform any reasoning steps. Instead it is asked to directly output a list, and with only four or five documents there are very few distinct possibilities.

This approach also has clear limitations:
* The number of documents is fixed across all implementations. If training uses four documents, scaling to twenty is difficult, while pairwise reranking does not have this constraint.
* Per image context size and cache in VLMs is huge.
* Order sensitivity can cause issues, even though the environment randomizes document order.
* The model must generate tokens in an autoregressive way instead of producing a single forward pass, so inference is slow.

Yet you can find the model here : UlrickBL/MultimodalQwen3LastRL-2B
And the training dataset here : UlrickBL/mm_reranker_rl_training

## RL for multimodality

To test this RL strategy, I used the GRPO trainer of the verifiers library that I had to adapt to vision model (especially with the pixel value trick and so on). You can find the implementation and PR here : https://github.com/PrimeIntellect-ai/verifiers/pull/409

