# Qwen3-VL-Reranker-2B VS Qwen3VLRerankerMM-2B : Lightweight vs Industrial Multimodal Reranking with Qwen 3

## Introduction

In November 2025, I finished training a multimodal reranker based on Qwen-3-VL-2B-Instruct. It achieved very strong performance, beating the SOTA Jina reranker m0 on most of my benchmarks while being trained with a very tight budget in both data and compute. This model is fast, performant, and efficient.

As of January 2026, Qwen has released its own version of a Qwen 3 VL 2B reranker.

In this blog post, I will walk through the differences in terms of training, architecture, and performance between the two models.

## Qwen-3-VL-2B-Reranker

![archi](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/QBnCXGLTOF6Z17PyISqy1.png)

Qwen built its reranker on top of the Qwen 3 LM Dense Instruct model and trained it using a binary classification task, following the same approach as for the Qwen 3 text reranker. The model takes the following inputs:

- A system prompt: `Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".`
- An instruction: the default one is `Given a search query, retrieve relevant candidates that answer the query.`, but it can be replaced with any other instruction.
- A query: it can be text, an image, or even a video.
- A document: it can be text, an image, or even a video.

The output is a probability that the document satisfies the given instruction.

![training](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/Zf7ctnlyEomBSSNI6Zyr6.png)

The model was trained on 40 million pairs, both synthetic and collected. We can assume that classic datasets such as ViDoRe training and or RealMMRag were used as the collected data for images. Synthetic data were used for videos, and both synthetic and collected data were used to augment images and text for both queries and documents.

This versatility makes the reranker truly multimodal, not simply cross-modal with text queries and image documents, as is the case for most so-called “multimodal” retrievers and rerankers that perform on ViDoRe V1, V2, and V3.

For the classification layer, Qwen constructs a single linear layer whose weight is:

$$
w = w_{\text{yes}} - w_{\text{no}}
$$

where w_yes and w_no are rows from the model’s LM head.

For a hidden state h, the layer outputs:

$$
s(h) = w^\top h = (w_{\text{yes}} - w_{\text{no}})^\top h = w_{\text{yes}}^\top h - w_{\text{no}}^\top h
$$

This is _exactly_:

$$
logit_{yes}(h)−logit_{no}(h)
$$

from the original language model head.

If you then apply a sigmoid:

$$
P(\text{yes} \mid h) = \sigma(s(h))
$$

you obtain the probability that “yes” beats “no” under the model’s own output embeddings.

This mechanism is directly derived from the base model.

As it is not described in the paper, we can assume that training is performed on all model weights, including the LM head.

## Key differences with Qwen3VLRerankerMM-2B

Both my model and Qwen’s rely on the same technique introduced in their previous text reranker:

- Use the yes and no logits to produce a pointwise probability.
- Start from an instruct base model and fine-tune it with SFT.
- Apply hard negative mining.

The main differences are:

- Dataset size: Qwen trained on 40 million samples, while I trained on 2,000 samples.
- Multimodality versus cross-modality: my objective was to build a model specialized for text queries and image documents only, whereas their model is highly versatile across modalities.
- Rephrasing: the model I trained uses both simple queries and rephrased queries from the RealMMRAG dataset.
- Training strategy: I used LoRA and kept the LM head frozen to prevent overfitting and preserve robustness.
- Maximum input resolution: 720 tokens per images for my model versus 1280 tokens per images for Qwen’s.

## Performances comparison

### Benchmark performances

In the Qwen Tech report, they compare the model to Jina Reranker m0 on several benchmark and dimensions :

![perf](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/BYVJQyjPTfeAArXtQoEa0.png)

If we focus on the 2B and Jina reranker m0 comon tasks are 3 tasks focusing on visual document queries :

- MMEB Image and VisDoc : VisDoc contains Vidore V1 and V2 dataset with some tasks like arxivqa

Doc:

![arxivqa](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/1NJg4bA85jWtnR0oqVSqV.png)

Query : What does the graph suggest about the 2-point correlation function as the value of X increases within the scaling limit at T>0?

- JinaVDR

Doc :

![jinavdr](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/Ii_5cLBg1RhNdpba0mKo2.png)

Query : What is the key algorithmic difference between temporal difference learning and adaptive dynamic programming?

- Vidore V3 :

Query : Is a variable name starting with a digit allowed in Python according to the document's content on naming conventions?

Doc :

![vidorev3](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/6YYoEPyZbLqO7CJ1qcACW.png)

Results are close, Qwen 3 outperforming Jina on 2 benchmarks.

I ran the 3 models on benchmark from my collection for reproducible results :

| **Dataset**                                                               | **Jina Reranker m0** | **Qwen3RerankerMM-2B** | Qwen3VL-Reranker-2B |
| ------------------------------------------------------------------------- | -------------------- | ---------------------- | ------------------- |
| UlrickBL/vidore_benchmark_2_esg_reports_human_labeled_v2_reranker_adapted | **0.851**            | 0.804                  | 0.806               |
| UlrickBL/REAL-MM-RAG_FinSlides_BEIR_reranker_adapted (rephrased level 1)  | 0.873                | **0.906**              | 0.90                |
| UlrickBL/vidore_benchmark_economics_reports_v2_reranker_adapted           | 0.735                | **0.813**              | 0.75                |
| UlrickBL/vidore_benchmark_arxivqa_reranker_adapted                        | 0.767                | 0.778                  | **0.87**            |

Surprisingly, the **Qwen3VL-Reranker-2B** model is very close to the lightly trained **Qwen3RerankerMM-2B** on human-labeled ESG data, where both are outperformed by **Jina Reranker m0**, and on RealMMRAG Finance Slides with level-one rephrasing.

On economics reports, **Qwen3RerankerMM-2B** outperforms both. However, on ArxivQA, probably the closest benchmark to the training distribution of the three models and the most semantically and keyword-oriented in terms of query and image alignment, **Qwen3VL-Reranker-2B** outperforms the others. This may be due to stronger semantic grounding and to the fact that the reranker was trained on both text and image reranking.

The difference could also stem from the input resolution. **Qwen3RerankerMM-2B** was trained and run with a maximum of 720 tokens per images, whereas **Qwen3VL-Reranker-2B** was trained and run with a maximum of 1280 tokens per images, almost twice the resolution.

However, I believe that keeping the same LM head and training only a small subset of parameters through LoRA can be beneficial and make the model more robust for out-of-distribution tasks, such as the economics report benchmark. This task was not explicitly targeted during training, yet it shows a significant improvement.

### Speed performances

In terms of speed, which is critical for RAG pipelines, I ran 50 inference passes on batches of 25 query/image pairs using the same data and the same GPUs (A100 40GB SXM4) with Flash Attention 2:

| **Qwen3VL-Reranker-2B** | 6 minutes 57 seconds |
| ----------------------- | -------------------- |
| **Jina Reranker m0**    | 2 minutes 12 seconds |
| **Qwen3RerankerMM-2B**  | 1 minute 41 seconds  |

This difference can likely be explained by:

- The number of max tokens used per images : a maximum reduction of 56% pixels
- The fact that the LM head is not replaced by an identity layer. As a result, the language model head is still executed during the forward pass. It represents roughly 14% of the model parameters, and therefore about 14% of the FLOPs in a single forward pass.
