

# Language Models' Perception of Their Knowledge Boundaries 

> A curated list of awesome papers about LMs' perception of their knowledge boundaries.  This repository will be continuously updated. If I missed any papers, feel free to open a PR to include them! And any feedback and contributions are welcome!

Knowing when LLMs lack knowledge enables them to express "I don't know" and trigger retrieval when they cannot provide correct answers. Consequently, much research focuses on **LLMs' perception of their knowledge boundaries—that is, whether they recognize what they know and what they don't.**

> :star: represents the same series of papers

## Contents

- [LMs' Perception of Their Knowledge Boundaries](#lms-perception-of-their-knowledge-boundaries)

  - [Survey or Foundation Papers](#survey-or-foundation-papers)
  - [White-box Investigation](#white-box-investigation)
    - [Training The Language Model](#training-the-language-model)
    - [Utilizing Internal States or Attention Weights](#utilizing-internal-states-or-attention-weights)
  - [Grey-box Investigation](#grey-box-investigation)
  - [Black-box Investigation](#black-box-investigation)

- [Adaptive RAG](#adaptive-rag)

  

## LMs' Perception of Their Knowledge Boundaries

These methods focus on determining whether the model can provide a correct answer but do not perform adaptive Retrieval-Augmented Generation (RAG).

### Survey or Foundation Papers

These papers are surveys or fairly comprehensive foundational studies.

- [Anthropic] [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221) *Saurav Kadavath et.al.* 11 Jul 2022

  > Study whether language models can evaluate the validity of their own claims and predict which questions they will be able to answer correctly

- [Survey] [Know Your Limits: A Survey of Abstention in Large Language Models](https://arxiv.org/abs/2407.18418) *Bingbing Wen et.al.* 25 Jul 2024

  > Introduce a framework to examine abstention from three perspectives: the query, the model, and human values. We organize the literature on abstention methods, benchmarks, and evaluation metrics using this framework, and discuss merits and limitations of prior work

- [Survey] [A Survey on the Honesty of Large Language Models](https://arxiv.org/abs/2409.18786) *Siheng Li et.al.* 27 Sep 2024

  > A survey on the honesty of LLMs, covering its clarification, evaluation approaches, and strategies for improvement	

- [Survey] [A Survey on Uncertainty Quantification of Large Language Models: Taxonomy,  Open Research Challenges, and Future Directions](https://arxiv.org/abs/2412.05563) *Ola Shorinwa et.al.* 7 Dec 2024

- [Survey] [Knowledge Boundary of Large Language Models: A Survey](https://arxiv.org/abs/2412.12472) *Moxin Li et.al.* 17 Dec 2024

- [Survey] [A Survey of Uncertainty Estimation Methods on Large Language Models](https://arxiv.org/pdf/2503.00172) *Zhiqiu Xia et.al.* 28 Feb 2025

### White-box Investigation

These methods require access to the full set of model parameters, such as for model training or using internal signals of the model.

#### Training The Language Model

- [EMNLP 2020, **Token-prob-based Confidence**] [Calibration of Pre-trained Transformers](https://arxiv.org/pdf/2003.07892) *Shrey Desai et.al.* 17 Mar 2020

  > Investigate calibration in pre-trained transformer models & in-domain and OOD settings. Find: 1) Pre-trained models are calibrated in-domain. 2) Label smooth is better that temperature scaling in OOD setting

- [TACL 2021, **Token-prob-based Confidence**] [How Can We Know When Language Models Know? On the Calibration of Language Models for Question Answering](https://arxiv.org/abs/2012.00955) *Zhengbao Jiang et.al.*  2 Dec 2020

  >1)Investigate calibration (answerr: not good) in generative language models (e.g., T5) in QA task (OOD settings). 2) Examine the effectiveness of some methods (fine-tuning, post-hoc probability modification, or adjustment of the predicted outputs or inputs)

- [TMLR 2022, **Token-prob-based & Verbalized Confidence**] [Teaching Models to Express Their Uncertainty in Words](https://arxiv.org/abs/2205.14334) *Stephanie Lin et.al.* 28 May 2022

  >  The first time a model has been shown to express calibrated uncertainty about its own answers in natural language. For testing calibration, we introduce the CalibratedMath suite of tasks

- [ACL 2023, **Token-prob-based Confidence**] [A Close Look into the Calibration of Pre-trained Language Models](https://arxiv.org/abs/2211.00151) *Yangyi Chen et.al.*  31 Oct 2022

  > Answer two questions: (1) Do PLMs learn to become calibrated in the training process? (No) (2) How effective are existing calibration methods? (learnable methods significantly reduce PLMs’ confidence in wrong predictions)

- [NeurIPS 2024, **Verbalized Confidence & Self-consistency**] [Alignment for Honesty](https://arxiv.org/abs/2312.07000) *Yuqing Yang et.al.* 12 Dec 2023

  > 1)Establishing a precise problem definition and defining “honesty” 2)introduce a flexible training framework which emphasize honesty without sacrificing performance on other tasks

- [​ :star: NeurIPS Safe Generative AI Workshop 2024, **Semantic Uncertainty**] [Fine-Tuning Large Language Models to Appropriately Abstain with Semantic Entropy](https://arxiv.org/abs/2410.17234) *Benedict Aaron Tjandra et.al.*  22 Oct 2024

  > Existing methods rely on the existence of ground-truth labels or are limited to short-form responses. This paper proposes fine-tuning using semantic entropy, an uncertainty measure derived from introspection into the model which does not require external labels.

- [NAACL 2024, **Verbalized Confidence**, **Outstanding Paper Award**] [R-Tuning: Instructing Large Language Models to Say ‘I Don’t Know’](https://arxiv.org/pdf/2311.09677) *Hanning Zhang et.al.* 7 Jun 2024

  > Proposeing a supervised finetuning and an unsupervised finetuning method:
  > 1)supervised:Add certainty tags to QA dataset based on model's answer correctness. Train the model to express uncertainty when not sure about its answer.
  > 2)unsupervised:Firstly, generate answer multiple times, and calculate entropy based on answer frequency(similar to semantic entropy but didn't use a NLI model).Secondly, separate high entropy data to 'uncertain' set and low entropy data to 'certain' set and finetune model.
  > Interestingly, unsupervised learning can improve both accuracy and calibration.

#### Utilizing Internal States or Attention Weights

These papers focus on determining the truth of a statement or the model’s ability to provide a correct answer by analyzing the model’s internal states or attention weights.  It usually involves using mathematical methods to extract features or training a lightweight MLP (Multi-Layer Perceptron).

- [EMNLP 2023 Findings] [The Internal State of an LLM Knows When It's Lying](https://arxiv.org/pdf/2304.13734) *Amos Azaria et.al.* 26 Apr 2023

  > LLM’s internal state can be used to reveal the truthfulness of statements（train a classifier using hidden states）

- [ICLR 2024] [Attention Satisfies: A Constraint-Satisfaction Lens on Factual Errors of Language Models](https://arxiv.org/abs/2309.15098) *Mert Yuksekgonul et.al.* 26 Sep 2023

  > Convert a QA problem into a constraint satisfaction problem (are the constraints in the question satisfied) and focus on the attention weights of each constraint when generating the first token.

- [EMNLP 2023] [The Curious Case of Hallucinatory (Un)answerability: Finding Truths in the Hidden States of Over-Confident Large Language Models](https://arxiv.org/abs/2310.11877) *Aviv Slobodkin* et.al. 18 Oct 2023

  >Investigate whether models already represent questions’ (un)answerablility when producing answers (Yes)

- [ICLR 2024] [INSIDE: LLMs’ internal states retain the power of hallucination detection](https://arxiv.org/abs/2402.03744) *Chao Chen et.al.* 6 Feb 2024

  > 1)Propose EigenScore metric using hidden states to better evaluate responses’ self-consistency and 2) Truncate extreme activations in the feature space, which helps identify the overconfident (consistent but wrong) hallucinations

- [ACL 2024 Findings, **MIND**] [Unsupervised Real-Time Hallucination Detection based on the Internal States of Large Language Models](https://arxiv.org/abs/2403.06448) *Weihang Su et.al.* 11 Mar 2024

  > Introduce MIND, an unsupervised training framework that leverages the internal states of LLMs for real-time hallucination detection without requiring manual annotations.

- [NAACL 2024] [On Large Language Models' Hallucination with Regard to Known Facts](https://arxiv.org/abs/2403.20009) *Che Jiang et.al.* 29 Mar 2024

  > Investigate the phenomenon of LLMs possessing correct answer knowledge yet still hallucinating from the perspective of inference dynamics

- [Arxiv, **FacLens**] [Hidden Question Representations Tell Non-Factuality Within and Across Large Language Models](https://arxiv.org/abs/2406.05328) *Yanling Wang et.al.* 8 Jun 2024 

  > Studies non-factuality prediction (NFP) before response generation and propose FacLens (train MLP) to enhance efficiency and transferability (across different models, the first in NFP) of NFP

- [Arxiv] [Towards Fully Exploiting LLM Internal States to Enhance Knowledge Boundary Perception](https://www.arxiv.org/abs/2502.11677) *Shiyu Ni et.al.* 17 Feb 2025

  > This work explores leveraging LLMs’ internal states to enhance their perception of knowledge boundaries from efficiency and risk perspectives. It focuses on: 1) The necessity of estimating model confidence after response generation. 2) Introducing Consistency-based Confidence Calibration ($C^3$), which evaluates confidence consistency through question reformulation. $C^3$ significantly improves LLMs’ ability to recognize their knowledge gaps.

### Grey-box Investigation

Need to access to the probability of generated tokens. Some methods also rely on the probability of generated tokens; however, since training is involved in the paper, they do not fall into this category.

- [ICML 2017, **Token-prob-based Confidence**] [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) *Chuan Guo et.al.* 14 Jun 2017

  > Investigate calibration in modern neural networks, propose ECE metric, propose enhance calibration via temperature


- [TACL 2022, **Verbalized Confidence**] [Reducing conversational agents’ overconfidence through linguistic calibration](https://aclanthology.org/2022.tacl-1.50/) *Sabrina J. Mielke et.al.* 30 Dec 2020

  > 1.Analyze to what extent SOTA chit-chat models are linguistically calibrated (poorly calibrated); 2.Train a much better correctness predictor directly from the chit-chat model’s representations. 3.Use this trained predictor within a controllable generation model which greatly improves the calibration of a SOTA chit-chat model.

- [ICLR 2023, **Token-prob-based Confidence & Self-consistency**] [Prompting GPT-3 To Be Reliable](https://arxiv.org/abs/2210.09150) *Chenglei Si et.al.* 17 Oct 2022

  > With appropriate prompts, GPT-3 is more reliable (both consistency-based and prob-based confidence estimation) than smaller-scale supervised models

- [​ :star: ICLR 2023 Spotlight, **Semantic Uncertainty**, **Token-prob-based Confidence & Self-consistency**] [Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation](https://arxiv.org/abs/2302.09664)  *Lorenz Kuhn et.al.* 19 Feb 2023

  > Introduce semantic entropy—an entropy which incorporates linguistic invariances created by shared meanings

- [ACL 2024, **Token-prob-based & Verbalized Confidence**] [Confidence Under the Hood: An Investigation into the Confidence-Probability Alignment in Large Language Models](https://arxiv.org/abs/2405.16282) *Abhishek Kumar et.al.* 25 May 2024

  > Investigate the alignment between LLMs' internal confidence and verbalized confidence


- [​ :star: Nature, **Semantic Uncertainty**] [Detecting hallucinations in large language models using semantic entropy](https://www.nature.com/articles/s41586-024-07421-0) *Sebastian Farquhar et.al.* 19 June 2024

  > The expension of [Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation](https://arxiv.org/abs/2302.09664) 

- [CCIR 2024, **Token-prob-based & Verbalized Confidence**] [Are Large Language Models More Honest in Their Probabilistic or Verbalized Confidence?](https://arxiv.org/abs/2408.09773) *Shiyu Ni et.al.* 19 Aug 2024

  > Conduct a comprehensive analysis and comparison of LLMs’ probabilistic perception and verbalized perception of their factual knowledge boundaries

### Black-box Investigation

These methods only require access to the model’s text output. 

- [EMNLP 2023, **Selfcheckgpt**, **Self-consistency**] [Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models](https://arxiv.org/abs/2303.08896) *Potsawee Manakul et.al.* 15 Mar 2023

  >  The first  to analyze model hallucination of general LLM responses, and is the first zero-resource hallucination detection solution that can be applied to black-box systems

- [ACL 2024, **Multi-LLM Collaboration**, **Outstanding paper award**] [Don't Hallucinate, Abstain: Identifying LLM Knowledge Gaps via Multi-LLM Collaboration](https://arxiv.org/abs/2402.00367)

  > Aim to identify LLM knowledge gaps and abstain from answering questions when knowledge gaps are present. Contribution: 1) A critical evaluation and typology of diverse existing methods 2) Propose two novel, robust multi-LLM collaboration methods to detect LLM knowledge gaps, COOPERATE and COMPETE
  
- [EMNLP 2023, **Token-prob-based & Verbalized Confidence**] [Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback](https://arxiv.org/abs/2305.14975) *Katherine Tian et.al.* 24 May 2023

  >Conduct a broad evaluation of methods for extracting confidence scores from RLHF-LMs

- [ACL 2023 Findings, **Verbalized Confidence**] [Do Large Language Models Know What They Don't Know?](https://arxiv.org/abs/2305.18153) *Zhangyue Yin et.al.* 29 May 2023

  >  Evaluate LLMs’ self-knowledge by assessing their ability to identify unanswerable or unknowable questions


- [EACL 2024 Findings, **Self-consistency**] [Do Language Models Know When They’re Hallucinating References](https://arxiv.org/abs/2305.18248) *Ayush Agrawal et.al.* 29 May 2023

  > Focus on hallucinated book and article references due to their frequent and easy-to-discern nature. Identify hallucinated references by asking a set of direct (yes/no questions to directly get the model's confidence) or indirect queries (ask for the authors of the generated reference) to the language model about the references.

- [ICLR 2024, **Self-consistency & Verbalized Confidence**] [Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs](https://arxiv.org/abs/2306.13063) *Miao Xiong et.al.*  22 Jun 2023

  > Explore black-box approaches for LLM uncertainty estimation. Define a systematic framework with three components: prompting strategies for eliciting verbalized confidence, sampling methods for generating multiple responses, and aggregation techniques for computing consistency

- [EMNLP 2023, **SAC3**, **Self-consistency & Multi-LLM Collaboration**] [SAC3: Reliable Hallucination Detection in Black-Box Language Models via Semantic-aware Cross-check Consistency](https://arxiv.org/abs/2311.01740) *Jiaxin Zhang et.al.* 3 Nov 2023 

  > Extend self-consistency across pertubed questions and different models

- [Arxiv] [Large Language Model Confidence Estimation via Black-Box Access](https://arxiv.org/abs/2406.04370) *Tejaswini Pedapati et.al.* 1 Jun 2024

  >  Engineer novel features and train a (interpretable) model (viz. logistic regression) on these features to estimate the confidence. Design different ways of manipulating the input prompt and produce values based on the variability of the answers for each such manipulation. We aver to these values as features

- [EMNLP 2024] [Calibrating the Confidence of Large Language Models by Eliciting Fidelity](https://arxiv.org/abs/2404.02655) *Mozhi Zhang et.al.* 3 April 2024

  >Decompose the language model's confidence for each choice into Uncertainty about the question and Fidelity to the answer. First, sample multiple times: 1.**Uncertainty**: The distribution of sampled answers. 2.**Fidelity**: Replace the selected answer with "all other options are wrong," then reselect, observing any changes. Repeat to assess fidelity to each answer. Finally, merge the two components.

## Adaptive RAG

These methods focus directly on the “when to retrieve”, designing strategies and evaluating their effectiveness in Retrieval-Augmented Generation (RAG).

- [ACL 2023 Oral, **Adaptive RAG**] [When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories (Adaptive RAG)](https://arxiv.org/abs/2212.10511) *Alex Mallen et.al.* 20 Dec 2022

  > Investigate 1) when we should and should not rely on LMs’ parametric knowledge and 2) how scaling and non-parametric memories (e.g., retrievalaugmented LMs) can help. Propose adaptive RAG based on entity popularity

- [EMNLP 2023, **FLARE**] [Active Retrieval Augmented Generation](https://arxiv.org/abs/2305.06983) *Zhengbao Jiang et.al.* 11 May 2023

  > Propose FLARE for long-form generation: Iteratively uses a prediction of the upcoming sentence to anticipate future content, which is then utilized as a query to retrieve relevant documents to regenerate the sentence if it contains low-confidence tokens

- [EMNLP 2023 Findings, **SKR**] [Self-Knowledge Guided Retrieval Augmentation for Large Language Models](https://arxiv.org/abs/2310.05002) *Yile Wang et.al.* 8 Oct 2023

  > Investigate eliciting the model’s ability to recognize what they know and do not know and propose Self-Knowledge guided Retrieval augmentation (SKR), which can let LLMs adaptively call retrieval

- [ICLR 2024 Oral, **Self-RAG**] [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511) *Akari Asai et.al.* 17 Oct 2023

  > Propose a new framework to train an arbitrary LM to learn to **retrieve, generate, and critique （via generating special tokens）**to enhance the factuality and quality of generations, without hurting the versatility of LLMs.

- [Arxiv, **Rowen**, **Enhanced SAC3**, **Self-consistency & Multi-LLM Collaboration**] [Retrieve Only When It Needs: Adaptive Retrieval Augmentation for Hallucination Mitigation in Large Language Models](https://arxiv.org/abs/2402.10612) *Hanxing Ding et.al.* 16 Feb 2024

  > Introduces Rowen which assesses the model’s uncertainty regarding the input query by evaluating the semantic **inconsistencies** in various responses generated **across different languages** or **models**.

- [ACL 2024 Findings, **Verbalized Confidence**, **Prompting Methods**] [When Do LLMs Need Retrieval Augmentation? Mitigating LLMs' Overconfidence Helps Retrieval Augmentation](https://arxiv.org/abs/2402.11457) *Shiyu Ni et.al.* 18 Feb 2024

  > 1)Quantitatively measure LLMs’ such ability and confirm their overconfidence 2) study how LLMs’ certainty about a question correlates with their dependence on external retrieved information 3)Propose several prompting methods to enhance LLMs’ perception of knowledge boundaries and show that they are effective in reducing overconfidence 4) equipped with these methods, LLMs can achieve comparable or even better performance of RA with much fewer retrieval calls.

- [Arxiv, **Position paper**] [Reliable, Adaptable, and Attributable Language Models with Retrieval](https://arxiv.org/abs/2403.03187) *Akari Asai et.al.* 5 Mar 2024

  > Advocate for retrieval-augmented LMs to replace parametric LMs as the next generation of LMs and propose a roadmap for developing general-purpose retrieval-augmented LMs

- [ACL 2024 Oral, **DRAGIN**, **Enhanced FLARE**] [DRAGIN: Dynamic Retrieval Augmented Generation based on the Information Needs of Large Language Models](https://arxiv.org/pdf/2403.10081) *Weihang Su et.al.* 15 Mar 2024

  > Propose Dragin, focusing on 1) when to retrieve: considers the LLM’s uncertainty about its own generated content, the influence of each token on subsequent tokens, and the semantic significance of each token and 2) what to retrieve: construct query using important words by leveraging the LLM’s self-attention across the entire context

- [Arxiv, **CtrlA**] [CtrlA: Adaptive Retrieval-Augmented Generation via Inherent Control](https://arxiv.org/abs/2405.18727) *Huanshuo Liu et.al.* 29 May 2024

  > This paper leverages the model's internal states to derive directions for honesty (where the output aligns with internal knowledge) and confidence. It enhances honesty by modifying internal representations and uses the confidence signal to detect retrieval timing.

- [EMNLP 2024 Findings, **UAR**] [Unified Active Retrieval for Retrieval Augmented Generation](https://arxiv.org/abs/2406.12534) *Qinyuan Cheng et.al.* 18 Jun 2024 

  > Propose Unified Active Retrieval (UAR)，consists of four orthogonal criteria for determining the retrieval timing: Intent-aware; Knowledge-aware; Time-Sensitive-aware; Selfa-aware

- [Arxiv, **SEAKR**] [SEAKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation](https://arxiv.org/abs/2406.19215) *Zijun Yao et.al.* 27 Jun 2024

  > Use hidden states of the last generated tokens to meauser LLMs' uncertainty and use this uncertainty to decide: when to retrieve, re-rank the retrieved documents, choose the reasoning strategy







