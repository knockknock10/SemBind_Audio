---

# 2.12 Candidate 3 — Vector Norm

The next observable quantity that naturally arises from an embedding is its **vector norm**.

Unlike cosine similarity, which measures the angular relationship between two vectors, the vector norm measures the magnitude (or length) of a single embedding.

Mathematically, the Euclidean norm of an embedding

\[
E_i=[e_1,e_2,\ldots,e_d]^T
\]

is

\[
\|E_i\|
=
\sqrt{
\sum_{k=1}^{d}
e_k^2
}.
\]

Since every embedding has a magnitude, one may naturally ask:

> **Can embeddings with larger magnitudes be considered more semantically important?**

At first glance, this appears plausible.

A larger magnitude might indicate that the neural network is more "confident" about a representation.

However, this interpretation is misleading.

---

## Intuition

The norm measures how far an embedding lies from the origin of the embedding space.

It does **not** describe

- contextual relevance,
- semantic necessity,
- linguistic importance,
- information content.

Consequently, two embeddings may possess identical semantic meanings while having different vector norms.

Likewise,

two completely unrelated embeddings may possess nearly identical magnitudes.

---

## Example 2.1

Suppose

\[
E_A
=
[3,4]
\]

and

\[
E_B
=
[-3,-4].
\]

Both vectors have norm

\[
5.
\]

Yet,

their directions are completely opposite.

If embedding geometry carries semantic information,

these vectors may represent entirely different concepts despite having identical norms.

Therefore,

vector magnitude alone provides very limited semantic evidence.

---

## Advantages

Vector norms possess several practical strengths.

### Advantage A1

Very inexpensive to compute.

---

### Advantage A2

Independent of neighboring embeddings.

---

### Advantage A3

May reflect certain properties of representation scaling.

---

## Limitations

However,

their limitations are considerably stronger.

### Limitation L1

No direct relationship with semantic meaning.

---

### Limitation L2

Sensitive to representation scaling.

---

### Limitation L3

Ignores contextual information completely.

---

### Limitation L4

Cannot distinguish semantic importance from representation magnitude.

---

## Conclusion

Although vector norms are mathematically simple,

they fail to satisfy the semantic requirements established earlier.

Consequently,

vector norms are unsuitable as the primary estimator of semantic importance.

---

# 2.13 Candidate 4 — Neighborhood Density

Another possible observable quantity is the **density of the local embedding neighborhood**.

Unlike cosine similarity,

which compares only two embeddings,

neighborhood density considers the local structure surrounding an embedding.

The intuition is straightforward.

If many embeddings occupy nearby locations,

the current embedding may belong to a common semantic region.

Conversely,

isolated embeddings may correspond to unique or unusual speech events.

---

## Definition

Let

\[
\mathcal{N}(E_i)
\]

denote the set of neighboring embeddings within a specified distance threshold.

The local density can be expressed conceptually as

\[
D_i
=
|\mathcal{N}(E_i)|.
\]

Larger values indicate denser regions of the embedding space.

---

## Motivation

Neighborhood density has been used successfully in

- clustering,
- anomaly detection,
- manifold learning,
- representation analysis.

Therefore,

it appears reasonable to investigate whether density also reflects semantic importance.

---

## Counterexample

Consider the repeated word

> "please"

appearing ten times.

Each occurrence may produce very similar embeddings,

creating an extremely dense neighborhood.

However,

repetition does not necessarily imply high semantic importance.

Conversely,

a rare word

> "defibrillator"

may occur only once,

producing a sparse neighborhood,

while contributing critically to the sentence meaning.

Therefore,

density measures **frequency of occurrence**, not necessarily semantic importance.

---

## Conclusion

Neighborhood density captures structural organization within the embedding space,

but structural frequency should not be confused with semantic necessity.

Accordingly,

density provides complementary evidence but cannot independently estimate semantic importance.

---

# 2.14 Candidate 5 — Entropy

Entropy is one of the most fundamental concepts in information theory.

Intuitively,

entropy measures uncertainty or unpredictability.

This naturally raises another research question.

> **Could embeddings with higher uncertainty correspond to higher semantic importance?**

Unfortunately,

the answer is not straightforward.

---

## Why Entropy Appears Attractive

Higher entropy often indicates

- greater variability,
- richer information,
- increased uncertainty.

This seems related to semantic complexity.

---

## The Problem

Semantic importance is not equivalent to uncertainty.

For example,

background noise may exhibit high entropy,

yet possess essentially no semantic value.

Conversely,

the word

> "No."

contains very little uncertainty,

but may completely determine the meaning of an entire conversation.

Thus,

uncertainty and semantic importance describe fundamentally different concepts.

---

## Conclusion

Entropy measures statistical uncertainty,

not semantic contribution.

Therefore,

entropy should not be used as a standalone semantic importance indicator.

---

# 2.15 Candidate 6 — Attention Scores

Modern transformer models compute attention weights between tokens or speech frames.

Because attention reflects information flow,

many researchers assume

> High attention implies high semantic importance.

Although tempting,

this assumption requires careful examination.

---

## Why Attention Appears Useful

Self-attention allows each embedding to interact with surrounding embeddings.

Consequently,

high attention weights often indicate strong contextual interactions.

---

## Important Limitation

Attention was designed to improve representation learning,

not to explain semantic importance.

High attention indicates that one representation influences another during computation.

It does **not** necessarily imply that the corresponding speech segment is semantically indispensable.

Indeed,

multiple studies have argued that

> **Attention is not explanation.**

Attention reflects computational dependency rather than semantic necessity.

---

## Conclusion

Attention contains valuable contextual information,

but should not be interpreted as a direct measure of semantic importance.

---

> [!IMPORTANT]
> Attention explains how the model processes information—not necessarily why one speech segment is semantically more important than another.

---

# 2.16 Comparative Analysis

The previous sections examined six candidate semantic indicators.

Each captures a different aspect of embedding geometry.

However,

none individually satisfies all of the desired properties.

The comparison is summarized below.

| Candidate | Semantic Evidence | Context Aware | Robust | Computational Cost | Sufficient Alone |
|-----------|-------------------|---------------|---------|--------------------|------------------|
| Cosine Similarity | High | Partial | High | Low | ✘ |
| Contextual Consistency | High | High | Moderate | Moderate | ✘ |
| Vector Norm | Low | No | Moderate | Very Low | ✘ |
| Neighborhood Density | Moderate | Partial | Moderate | Moderate | ✘ |
| Entropy | Low | No | Low | Moderate | ✘ |
| Attention | Moderate | High | High | High | ✘ |

Several important observations emerge.

First,

every indicator captures useful information.

Second,

every indicator also ignores important semantic properties.

Third,

no candidate satisfies all six evaluation criteria introduced earlier.

---

# 2.17 Why a Hybrid Estimator Is Necessary

The comparison above reveals a fundamental insight.

Semantic importance is a complex latent property.

Expecting a single observable quantity to capture every aspect of semantic contribution is unrealistic.

Instead,

each observable quantity provides **partial evidence**.

This observation motivates the central hypothesis of SemBind-Audio.

---

## Research Hypothesis H2

Semantic importance should be estimated using multiple complementary observable quantities rather than a single embedding-space property.

---

Conceptually,

suppose

\[
O_1,O_2,\ldots,O_m
\]

represent different observable indicators.

Instead of constructing

\[
\hat{S}_i
=
f(O_1),
\]

SemBind-Audio proposes a more general framework,

\[
\hat{S}_i
=
f(O_1,O_2,\ldots,O_m).
\]

Notice that

we have **not yet defined**

the exact mathematical form of

\[
f.
\]

That derivation belongs to later chapters.

At this stage,

our conclusion is simply that

multiple complementary indicators are theoretically preferable to relying on any single observable quantity.

---

> [!OUR CONTRIBUTION]
> Existing watermarking methods often adopt one heuristic (energy, masking threshold, or perceptual model) for region selection. SemBind-Audio instead formulates region selection as a **multi-evidence estimation problem**, where semantic importance is inferred from complementary embedding-space observations.

---

# 2.18 Chapter Summary

This chapter investigated six candidate observable indicators that may provide evidence about semantic importance.

Rather than immediately constructing a semantic score,

each candidate was analyzed independently using the evaluation criteria established at the beginning of the chapter.

The analysis demonstrated that

- cosine similarity captures geometric alignment,
- contextual consistency captures local semantic flow,
- vector norms capture representation magnitude,
- neighborhood density captures local embedding structure,
- entropy captures statistical uncertainty,
- attention captures computational dependency.

Although each indicator contributes useful information,

none alone provides a complete estimate of semantic importance.

Consequently,

semantic importance should be treated as a latent variable inferred from multiple complementary sources of evidence.

This conclusion establishes the theoretical justification for the hybrid estimation framework developed in subsequent chapters.

---

# Key Takeaways

1. Semantic importance is more complex than any single geometric property.

2. Every observable indicator provides only partial evidence.

3. Cosine similarity alone is insufficient.

4. Context alone is insufficient.

5. Vector norm, density, entropy, and attention also exhibit important limitations.

6. A principled semantic estimator should integrate multiple complementary indicators.

7. The next chapter introduces the concept of a **Global Audio Representation**, which provides the reference point against which semantic similarity will be evaluated.

---

# Reviewer Questions

1. Why is cosine similarity insufficient for estimating semantic importance?

2. Why does vector norm fail as a semantic indicator?

3. What is the difference between neighborhood density and semantic importance?

4. Why should entropy not be interpreted as semantic importance?

5. Why is "attention is not explanation" relevant to this research?

6. Why does every candidate indicator capture only partial evidence?

7. What motivates the hybrid estimation hypothesis?

8. Why is semantic importance fundamentally different from observable embedding properties?

---

# References

## [R1] Bishop, C. M. *Pattern Recognition and Machine Learning.*

**Used for**

- Statistical estimation
- Feature spaces
- Latent variable reasoning

---

## [R2] Goodfellow, I., Bengio, Y., & Courville, A. *Deep Learning.*

**Used for**

- Representation learning
- Distributed representations
- Learned feature spaces

---

## [R3] Vaswani, A., et al. (2017). *Attention Is All You Need.*

**Used for**

- Transformer architecture
- Self-attention mechanism

**Not interpreted as evidence that attention directly measures semantic importance.**

---

## [R4] Jain, S., & Wallace, B. C. (2019). *Attention is not Explanation.*

**Used for**

- Critical discussion of attention as an explanatory mechanism.

---

# Bridge to Chapter 03

Throughout this chapter, we repeatedly discussed comparing an embedding to the **overall semantic content** of the audio. However, one fundamental question remains unanswered:

> **What exactly represents the "overall semantic content" of an audio recording?**

Individual embeddings describe only local speech segments. To compare each embedding against the audio as a whole, we first require a single vector that summarizes the entire recording.

The next chapter derives this concept formally by introducing the **Global Audio Representation**, explains why such a reference vector is necessary, investigates multiple pooling strategies (mean, max, median, attention pooling), and mathematically justifies the choice adopted in SemBind-Audio.

# SemBind-Audio Research Mathematics Notebook

# Chapter 03

# Global Audio Representation

---

# Learning Objectives

After completing this chapter, the reader should be able to:

- Understand why a global audio representation is required.
- Explain the difference between local and global representations.
- Define the mathematical properties required for a global representation.
- Compare different pooling strategies.
- Justify the choice of mean pooling in the SemBind-Audio framework.
- Understand the limitations of global representations.

---

# Prerequisites

Before reading this chapter, the reader should understand:

- Representation learning
- Embeddings
- Latent spaces
- Semantic importance estimation
- Candidate semantic indicators

introduced in Chapters 00, 01, and 02.

---

# Chapter Overview

The previous chapter established that semantic importance cannot be measured directly and must instead be estimated using observable properties of contextual embeddings.

One promising observable quantity is the similarity between an individual embedding and the overall semantic content of the audio recording.

However, this immediately raises an important question.

> **What mathematically represents the overall semantic content of an audio recording?**

At present, we possess only a sequence of local contextual embeddings,

\[
E_1,E_2,\ldots,E_N,
\]

where each embedding represents only a short speech segment.

No embedding individually represents the complete recording.

Therefore, before constructing any semantic importance estimator, we must first derive a single vector capable of summarizing the entire audio.

This chapter develops that representation from first principles.

Rather than immediately adopting an averaging operation, we first establish the mathematical requirements of a global representation, investigate multiple candidate pooling strategies, analyze their strengths and limitations, and finally justify the representation adopted by SemBind-Audio.

---

# 3.1 Motivation

Consider an audio recording consisting of several spoken sentences.

After processing the recording using HuBERT, we obtain

\[
N
\]

contextual embeddings,

\[
E_1,E_2,\ldots,E_N,
\]

where

\[
E_i\in\mathbb{R}^{768}.
\]

Each embedding represents only one local speech segment.

Suppose we now ask the following question.

> **How semantically important is the \(i^{\text{th}}\) embedding with respect to the entire audio recording?**

This question cannot be answered by examining

\[
E_i
\]

alone.

Semantic importance is inherently a relative concept.

An embedding may appear important when viewed in isolation but may contribute very little to the overall meaning of the complete recording.

Conversely, another embedding may appear ordinary by itself while carrying crucial semantic information within the broader context of the audio.

Therefore, evaluating semantic importance requires a reference representation describing the audio as a whole.

This motivates the introduction of a **Global Audio Representation**.

---

> [!IMPORTANT]
> Semantic importance is not an intrinsic property of an individual embedding. It is defined relative to the semantic content of the complete audio recording.

---

# 3.2 Local Representation vs Global Representation

To understand the need for a global representation, it is useful to distinguish between local and global information.

A local representation summarizes only a small portion of the audio.

A global representation summarizes the recording as a whole.

Within the SemBind-Audio framework,

each HuBERT embedding

\[
E_i
\]

is a local representation.

The Global Audio Representation,

denoted by

\[
G,
\]

is intended to summarize the semantic information contained across all embeddings.

The relationship may be visualized conceptually as

```
Input Audio
      │
      ▼
HuBERT Encoder
      │
      ▼
E₁   E₂   E₃   ...   Eₙ
 \    |    |         /
  \   |    |        /
   \  |    |       /
    ───────────────
           │
           ▼
           G
```

Notice that

\[
G
\]

does not replace the individual embeddings.

Instead,

it provides a common semantic reference against which every embedding can later be compared.

---

# 3.3 Why Is a Global Representation Necessary?

Suppose we wish to compute the semantic relevance of

\[
E_i.
\]

One possible idea is to compare

\[
E_i
\]

with another embedding,

for example,

\[
E_{i+1}.
\]

Although this comparison reveals local similarity,

it does not indicate whether

\[
E_i
\]

is important to the overall recording.

Similarly,

comparing

\[
E_i
\]

with only its neighboring embeddings captures local context but ignores long-range semantic relationships.

To evaluate semantic contribution relative to the complete recording, every embedding should be compared against a representation of the entire audio.

Consequently,

we require a vector

\[
G
\]

that satisfies

\[
G\in\mathbb{R}^{768},
\]

and summarizes the semantic content of

\[
\{E_1,E_2,\ldots,E_N\}.
\]

Once such a representation has been constructed,

semantic similarity can be evaluated between

\[
E_i
\]

and

\[
G.
\]

This comparison forms one of the primary components of the semantic importance estimator developed in later chapters.

---

## Definition 3.1 — Global Audio Representation

A **Global Audio Representation** is a vector that summarizes the information contained across all contextual embeddings of an audio recording while remaining within the same embedding space.

Mathematically,

\[
G\in\mathbb{R}^{768},
\]

where

\[
G
\]

represents the complete recording rather than any individual speech segment.

Unlike

\[
E_i,
\]

which represents local information,

\[
G
\]

represents global semantic information.

---

> [!FACT]
> The Global Audio Representation serves as a reference vector for measuring the semantic relationship between individual embeddings and the overall recording. It is not itself a semantic importance score.

---

# 3.4 Design Requirements

Before deciding how to construct

\[
G,
\]

we first establish the properties that a suitable Global Audio Representation should satisfy.

### Requirement R1 — Global Coverage

The representation should summarize information from the entire recording rather than a single speech segment.

---

### Requirement R2 — Representation Consistency

The resulting vector should remain within the same embedding space as the individual HuBERT embeddings.

This ensures that similarity measures such as cosine similarity remain mathematically valid.

---

### Requirement R3 — Computational Efficiency

The representation should be computable efficiently even for long recordings.

---

### Requirement R4 — Robustness

Small perturbations in one embedding should not drastically alter the global representation.

---

### Requirement R5 — Compatibility

The representation should integrate naturally with the semantic importance estimator developed in later chapters.

---

### Requirement R6 — No Circular Reasoning

The construction of

\[
G
\]

must not require prior knowledge of semantic importance.

Otherwise,

the framework would assume the very quantity it is attempting to estimate.

---

> [!IMPORTANT]
> Requirement R6 is particularly critical. Any method requiring semantic importance weights before semantic importance has been estimated introduces circular reasoning and is therefore unsuitable.


---

# 3.5 Candidate 1 — Mean Pooling

Having established the requirements for a Global Audio Representation, we now investigate the first candidate: **Mean Pooling**.

Mean pooling is one of the most widely used aggregation techniques in representation learning because of its simplicity, efficiency, and mathematical stability.

However, rather than adopting it immediately, we first derive it from first principles.

---

## Motivation

Suppose an audio recording is represented by

\[
N
\]

HuBERT embeddings,

\[
E_1,E_2,\ldots,E_N,
\]

where

\[
E_i\in\mathbb{R}^{768}.
\]

Our objective is to construct a single vector

\[
G
\]

that summarizes the complete recording.

The simplest idea is to allow every embedding to contribute equally.

If every embedding contributes equally, then each should receive the same weight.

Since there are

\[
N
\]

embeddings,

each embedding receives weight

\[
\frac{1}{N}.
\]

The resulting representation becomes

\[
\boxed{
G
=
\frac{1}{N}
\sum_{i=1}^{N}
E_i
}
\]

This operation is called **mean pooling**.

---

## Why Divide by N?

An obvious question arises.

Why not simply compute

\[
\sum_{i=1}^{N}E_i
\]

instead?

The answer is straightforward.

Without division,

the magnitude of

\[
G
\]

would increase as the recording becomes longer.

Consequently,

two recordings containing identical semantic content but different durations would produce vectors with substantially different magnitudes.

Dividing by

\[
N
\]

normalizes the representation,

making it independent of recording length.

Thus,

the arithmetic mean preserves representational consistency across recordings of different durations.

---

## Interpretation

Each embedding contributes equally to the global representation.

No embedding is assumed to be more important than another.

This is an intentional design choice.

At this stage,

semantic importance has not yet been estimated.

Assigning unequal weights would therefore require information that is not yet available.

Consequently,

equal weighting provides the least biased aggregation strategy.

---

## Advantages

Mean pooling satisfies several of the requirements established earlier.

### Advantage A1 — Global Coverage

Every embedding contributes to

\[
G.
\]

No speech segment is ignored.

---

### Advantage A2 — Same Embedding Space

Since every

\[
E_i
\]

belongs to

\[
\mathbb{R}^{768},
\]

their arithmetic mean also belongs to

\[
\mathbb{R}^{768}.
\]

Therefore,

\[
G
\]

remains directly comparable with every embedding.

---

### Advantage A3 — Computational Efficiency

Computing

\[
G
\]

requires only one pass through the embeddings.

The computational complexity is

\[
O(Nd),
\]

where

- \(N\) is the number of embeddings,
- \(d=768\) is the embedding dimension.

This makes mean pooling highly scalable.

---

### Advantage A4 — Stable Representation

Small perturbations in a single embedding influence only a small portion of the overall average.

Consequently,

mean pooling is relatively robust against local noise.

---

## Limitations

Despite these strengths,

mean pooling also has important limitations.

### Limitation L1

Every embedding contributes equally,

regardless of its semantic importance.

---

### Limitation L2

Rare but highly informative speech segments may become diluted within the average.

---

### Limitation L3

Temporal ordering is ignored.

The vectors

\[
(E_1,E_2,E_3)
\]

and

\[
(E_3,E_2,E_1)
\]

produce the same average.

---

> [!IMPORTANT]
> Mean pooling is **not** claimed to produce the perfect global representation.
>
> It is selected because it satisfies the design requirements while avoiding unjustified assumptions regarding semantic importance.

---

# 3.6 Candidate 2 — Max Pooling

Another possible aggregation strategy is **Max Pooling**.

Instead of averaging,

the largest value in every embedding dimension is selected.

Mathematically,

the

\[
j^{th}
\]

component of

\[
G
\]

is

\[
G_j
=
\max_i(E_{ij}).
\]

The resulting vector consists of the maximum value observed in every dimension.

---

## Motivation

The intuition behind max pooling is that the strongest activation may correspond to the most informative feature.

Consequently,

important semantic characteristics may survive the pooling process.

---

## Advantages

### Advantage A1

Highlights strong activations.

---

### Advantage A2

Simple to compute.

---

### Advantage A3

Often performs well in image processing.

---

## Limitations

Unfortunately,

max pooling exhibits several important weaknesses.

### Limitation L1

Different dimensions may originate from different embeddings.

Consequently,

the resulting vector may never have existed within the embedding space.

---

### Limitation L2

The strongest activation does not necessarily correspond to the most important semantic contribution.

---

### Limitation L3

Very sensitive to outliers.

A single noisy embedding can dominate an entire dimension.

---

### Limitation L4

Produces unstable representations for speech.

Small changes in one embedding may dramatically alter the pooled representation.

---

## Conclusion

Although max pooling preserves strong activations,

it violates several requirements established earlier.

Therefore,

it is not selected for SemBind-Audio.

---

# 3.7 Candidate 3 — Median Pooling

Median pooling replaces the arithmetic mean with the median computed independently for every embedding dimension.

The median is widely known for its robustness against extreme values.

---

## Motivation

If a small number of embeddings contain unusually large values,

the arithmetic mean may become biased.

The median is considerably less sensitive to such outliers.

---

## Advantages

### Advantage A1

Robust against extreme values.

---

### Advantage A2

Stable under noisy observations.

---

### Advantage A3

Simple mathematical formulation.

---

## Limitations

Despite these advantages,

median pooling also suffers from several shortcomings.

### Limitation L1

The resulting vector does not necessarily correspond to any meaningful speech representation.

---

### Limitation L2

The median suppresses informative deviations that may actually carry important semantic information.

---

### Limitation L3

Semantic relationships between embeddings are ignored.

---

## Conclusion

Median pooling provides statistical robustness,

but statistical robustness alone does not imply semantic representativeness.

Accordingly,

median pooling is not adopted.

---

# 3.8 Candidate 4 — Attention Pooling

At first glance,

attention pooling appears to be the most sophisticated aggregation strategy.

Instead of assigning equal weights,

each embedding receives an importance weight

\[
\alpha_i.
\]

The global representation becomes

\[
G
=
\sum_{i=1}^{N}
\alpha_iE_i,
\]

subject to

\[
\sum_{i=1}^{N}\alpha_i=1.
\]

---

## Why Does Attention Pooling Appear Attractive?

Unlike mean pooling,

attention pooling allows more informative embeddings to contribute more strongly to the global representation.

This appears consistent with our objective of identifying semantically important regions.

---

## The Circular Reasoning Problem

However,

attention pooling introduces a fundamental theoretical difficulty.

To compute

\[
\alpha_i,
\]

the model must already know which embeddings are important.

But determining semantic importance is precisely the research problem addressed by SemBind-Audio.

Consequently,

attention pooling assumes the answer before solving the problem.

This creates **circular reasoning**.

---

### Circular Dependency

The dependency can be illustrated as follows.

```
Need semantic importance
        │
        ▼
Compute attention weights
        │
        ▼
Construct global representation
        │
        ▼
Estimate semantic importance
```

The estimator depends upon itself.

Such a formulation is mathematically invalid for our framework.

---

> [!IMPORTANT]
> Attention pooling is rejected **not because it is ineffective**, but because it violates Requirement R6 by requiring prior importance information that is unavailable during estimator construction.

---

# 3.9 Comparative Analysis

The four candidate pooling strategies are summarized below.

| Method | Global Coverage | Computational Efficiency | Robustness | Circular Reasoning | Selected |
|---------|-----------------|--------------------------|------------|-------------------|----------|
| Mean Pooling | ✔ | ✔ | ✔ | ✘ | ✔ |
| Max Pooling | Partial | ✔ | ✘ | ✘ | ✘ |
| Median Pooling | Partial | ✔ | Moderate | ✘ | ✘ |
| Attention Pooling | ✔ | Moderate | ✔ | ✔ | ✘ |

Several important observations emerge.

Mean pooling satisfies every design requirement established earlier.

Attention pooling provides greater flexibility,

but introduces circular reasoning.

Max pooling emphasizes isolated activations rather than holistic representations.

Median pooling improves statistical robustness but sacrifices semantic representativeness.

Therefore,

mean pooling provides the most appropriate balance between theoretical soundness and practical applicability.

---

# 3.10 Final Global Audio Representation

Based on the preceding analysis,

SemBind-Audio adopts the following Global Audio Representation.

\[
\boxed{
G
=
\frac{1}{N}
\sum_{i=1}^{N}
E_i
}
\]

where

- \(E_i\in\mathbb{R}^{768}\),
- \(N\) denotes the number of embeddings,
- \(G\in\mathbb{R}^{768}\).

This representation serves as the semantic reference vector used throughout the remainder of the framework.

---

# 3.11 Mathematical Properties

The selected Global Audio Representation possesses several desirable mathematical properties.

### Property P1

\[
G\in\mathbb{R}^{768}.
\]

Therefore,

all cosine similarity computations remain mathematically valid.

---

### Property P2

The computational complexity is

\[
O(Nd).
\]

---

### Property P3

The representation is invariant to recording length after normalization.

---

### Property P4

Every embedding contributes equally,

preventing premature semantic assumptions.

---

### Property P5

The representation is directly compatible with the semantic importance estimator developed in the next chapter.

---

# 3.12 Chapter Summary

This chapter established the mathematical foundation for representing an entire audio recording using a single vector.

Four candidate aggregation strategies—mean pooling, max pooling, median pooling, and attention pooling—were analyzed according to the design requirements defined at the beginning of the chapter.

Mean pooling was selected because it provides a mathematically consistent, computationally efficient, and unbiased global representation while avoiding circular reasoning.

The resulting Global Audio Representation,

\[
G,
\]

serves as the semantic reference against which individual embeddings will be compared in the next chapter.

---

# Bridge to Chapter 04

The Global Audio Representation has now been formally defined.

The next question becomes:

> **How should an individual embedding be compared with this global representation?**

To answer this question, the next chapter derives **Cosine Similarity** from first principles, explains why it is the most appropriate measure of geometric alignment in embedding spaces, and establishes it as the first component of the SemBind-Audio semantic importance estimator.

# SemBind-Audio Research Mathematics Notebook

# Chapter 04

# Global Semantic Alignment using Cosine Similarity

---

# Learning Objectives

After completing this chapter, the reader should be able to:

- Understand why cosine similarity is appropriate for embedding spaces.
- Derive cosine similarity from first principles.
- Understand the geometric interpretation of cosine similarity.
- Explain why cosine similarity is preferred over Euclidean distance.
- Define Global Semantic Alignment mathematically.
- Understand the limitations of cosine similarity.

---

# Prerequisites

Before reading this chapter, the reader should understand:

- Vector spaces
- Embeddings
- Global Audio Representation
- Representation learning

introduced in Chapters 00–03.

---

# Chapter Overview

The previous chapter introduced the Global Audio Representation

\[
G.
\]

This vector summarizes the semantic information contained within the entire audio recording.

However,

having constructed

\[
G,
\]

we still cannot estimate semantic importance.

A second mathematical question must now be answered.

> **How similar is each embedding to the global semantic content of the audio?**

This chapter derives a mathematical measure capable of answering that question.

Rather than adopting cosine similarity because it is widely used,

we derive it from vector geometry,

justify why it is appropriate for learned embedding spaces,

analyze its properties,

and explain its role within the SemBind-Audio framework.

---

# 4.1 Motivation

Suppose

\[
E_i
\]

represents one speech segment,

while

\[
G
\]

represents the complete recording.

Intuitively,

if

\[
E_i
\]

closely aligns with

\[
G,
\]

the corresponding speech segment may contribute substantially to the overall semantic content.

Conversely,

if

\[
E_i
\]

deviates considerably from

\[
G,
\]

its semantic relationship to the overall recording may be weaker.

Therefore,

the first observable quantity investigated in the semantic importance estimator is the similarity between

\[
E_i
\]

and

\[
G.
\]

---

> [!IMPORTANT]
> Cosine similarity does **not** directly measure semantic importance. Instead, it measures geometric alignment between an embedding and the Global Audio Representation.

---

# 4.2 Why Not Euclidean Distance?

Before selecting cosine similarity,

we first examine another natural candidate:

Euclidean distance.

Given

\[
E_i
\]

and

\[
G,
\]

their Euclidean distance is

\[
d(E_i,G)
=
\sqrt{
\sum_{k=1}^{768}
(E_{ik}-G_k)^2
}.
\]

This quantity measures the straight-line distance between two vectors.

At first glance,

Euclidean distance appears perfectly reasonable.

However,

deep embedding spaces possess properties that make Euclidean distance less suitable.

---

## Limitation 1

Euclidean distance depends upon vector magnitude.

Two vectors pointing in exactly the same direction but having different magnitudes may appear far apart.

---

## Limitation 2

Transformer embeddings primarily encode information through geometry rather than absolute magnitude.

Consequently,

direction is generally more informative than length.

---

## Limitation 3

Embeddings extracted from different recordings may exhibit different scales.

Euclidean distance is sensitive to such scaling.

---

> [!FACT]
> Modern representation learning generally compares embeddings using angular similarity rather than Euclidean distance because direction carries more meaningful information than magnitude.

---

# 4.3 Geometric Interpretation

Consider two vectors,

\[
A
\]

and

\[
B.
\]

Suppose both originate from the origin of the embedding space.

These vectors form an angle

\[
\theta.
\]

If

\[
\theta=0^\circ,
\]

the vectors point in exactly the same direction.

If

\[
\theta=90^\circ,
\]

they are orthogonal.

If

\[
\theta=180^\circ,
\]

they point in opposite directions.

Notice that this interpretation depends only upon direction,

not magnitude.

Therefore,

angular similarity naturally captures geometric alignment.
