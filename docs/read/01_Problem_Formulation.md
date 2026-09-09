# SemBind-Audio Research Mathematics Notebook

# Chapter 01

# Problem Formulation

---

# Learning Objectives

After completing this chapter, the reader should be able to:

- Understand the research problem addressed by SemBind-Audio.
- Formulate the watermarking problem mathematically.
- Distinguish between observable and latent variables.
- Understand why semantic importance cannot be directly measured.
- Define the objective of semantic importance estimation.
- Understand the assumptions underlying the proposed framework.
- Recognize the scope and limitations of the formulation.

---

# Prerequisites

Before reading this chapter, the reader should understand:

- Representations
- Embeddings
- Latent space
- HuBERT embeddings
- Distributed representations

introduced in **Chapter 00 – Foundations**.

---

# Chapter Overview

The previous chapter introduced the mathematical foundations required to represent speech using contextual embeddings.

However, representing speech is only the first step.

The central research question of SemBind-Audio is considerably more challenging.

Rather than asking

> "How can speech be represented?"

our research asks

> **"Which parts of a speech representation are semantically important, and which parts can safely carry a watermark without significantly affecting semantic content?"**

Answering this question requires a precise mathematical formulation.

Unlike many classical watermarking systems that rely on acoustic energy or psychoacoustic masking, our framework attempts to estimate **semantic importance** from learned speech representations.

The difficulty is that semantic importance is **not directly observable**.

Consequently, this chapter formulates semantic importance as a **latent variable** that must be estimated indirectly from observable quantities.

This distinction forms the theoretical foundation of the entire SemBind-Audio framework.

---

# 1.1 Research Motivation

Digital audio watermarking aims to embed ownership or authentication information into an audio signal while preserving perceptual quality and robustness.

Traditional watermarking algorithms generally determine embedding locations using signal-processing criteria such as

- energy,
- spectral masking,
- psychoacoustic thresholds,
- frequency bands,
- temporal masking.

These approaches attempt to minimize perceptual distortion.

However, they rarely consider whether a particular speech region carries **important semantic information**.

For example, consider the sentence

> "The patient is **not** allergic to penicillin."

Suppose a watermark introduces distortion into the word

> "not"

Although the overall acoustic energy of this word may be relatively small, altering it changes the semantic meaning of the entire sentence.

Consequently,

perceptual importance and semantic importance are not necessarily identical.

This observation motivates the central hypothesis of SemBind-Audio.

> Watermark embedding should be guided not only by acoustic characteristics but also by semantic importance.

To investigate this idea rigorously, we must first define what semantic importance means mathematically.

---

> [!RESEARCH HYPOTHESIS H1]
> Embedding watermark information into semantically less important regions can preserve semantic integrity more effectively than selecting embedding regions solely using low-level acoustic properties.

---

# 1.2 Problem Statement

Let

\[
X
\]

denote an input speech signal.

Rather than operating directly on the waveform,

SemBind-Audio first transforms the signal into contextual embeddings using a pretrained speech representation model.

Assume that the embedding extractor produces

\[
N
\]

frame-level embeddings,

\[
E_1,E_2,\ldots,E_N,
\]

where

\[
E_i\in\mathbb{R}^{768},
\]

for

\[
i=1,2,\ldots,N.
\]

Each embedding summarizes contextual information corresponding to one short segment of the original speech signal.

The objective of SemBind-Audio is **not** to modify every embedding.

Instead,

we seek to identify which embeddings are

- semantically critical,
- moderately important,
- relatively unimportant.

Watermark information should preferably be embedded into embeddings that have lower semantic importance while avoiding highly informative regions whenever possible.

Therefore,

our problem becomes one of **importance estimation**.

---

## Mathematical Formulation

Given

\[
\mathcal{E}
=
\{E_1,E_2,\ldots,E_N\},
\]

our objective is to construct a function

\[
f:
\mathbb{R}^{768}
\rightarrow
\mathbb{R},
\]

such that

\[
S_i
=
f(E_i),
\]

where

\[
S_i
\]

denotes the estimated semantic importance of the

\[
i^{\text{th}}
\]

embedding.

Higher values of

\[
S_i
\]

indicate greater semantic importance,

while lower values indicate greater suitability for watermark embedding.

At this stage,

the function

\[
f(\cdot)
\]

is unknown.

Determining an appropriate mathematical formulation for

\[
f
\]

constitutes the primary research problem addressed in this notebook.

---

> [!IMPORTANT]
> Defining the semantic importance function is **the core mathematical challenge** of SemBind-Audio. Every subsequent chapter contributes toward constructing an estimator for this unknown function.

---

# 1.3 Observable and Latent Variables

Before attempting to estimate semantic importance,

we must distinguish between quantities that are directly observable and those that are hidden.

This distinction is fundamental in statistics, machine learning, and probabilistic modeling.

---

## Observable Variables

An observable variable is any quantity that can be directly measured or computed from the available data.

Within the SemBind-Audio framework,

examples include

- waveform samples,
- spectrogram values,
- HuBERT embeddings,
- cosine similarities,
- vector norms,
- distances between embeddings.

These quantities can be computed algorithmically without ambiguity.

---

## Latent Variables

A latent variable is fundamentally different.

It represents a property that exists conceptually but cannot be measured directly.

Examples include

- intelligence,
- emotion,
- intent,
- topic relevance,
- semantic importance.

Such quantities are not directly available from the data.

Instead,

they must be inferred indirectly from observable evidence.

---

## Definition 1.1 — Latent Variable

A **latent variable** is an unobserved quantity whose value must be estimated using observable variables and an appropriate mathematical model.

---

In SemBind-Audio,

semantic importance belongs to this category.

There exists no sensor,

algorithm,

or annotation process that directly outputs

> "This embedding has semantic importance 0.83."

Therefore,

semantic importance cannot be measured directly.

Instead,

it must be inferred.

This observation immediately changes the nature of our problem.

We are **not computing** semantic importance.

We are **estimating** semantic importance.

---

> [!FACT]
> The distinction between computation and estimation is crucial. Observable quantities are computed directly from data, whereas latent variables require inference through mathematical models.

---

# 1.4 Why Semantic Importance Is a Latent Variable

The previous section established that semantic importance cannot be observed directly. In this section, we investigate **why** this is true.

At first glance, one might assume that semantic importance can simply be measured from the speech signal itself. However, this assumption is incorrect because semantic importance is **not a physical property** of the signal.

Consider an audio waveform.

Mathematically, the waveform is simply a sequence of sampled amplitudes,

\[
X=[x_1,x_2,\ldots,x_T].
\]

Each sample describes the air pressure recorded by the microphone at a particular instant.

The waveform explicitly contains information such as

- amplitude,
- frequency,
- duration,
- phase,
- spectral content.

These are **physical properties** and can therefore be measured directly.

Semantic importance is fundamentally different.

Questions such as

- Which word is most important?
- Which speech segment carries the greatest semantic meaning?
- Which region can tolerate watermark embedding?

cannot be answered by directly inspecting waveform amplitudes.

Instead, answering these questions requires **interpreting** the speech.

Interpretation is an inference process rather than a measurement process.

Consequently, semantic importance belongs to the class of latent variables.

---

## Example 1.1

Consider the sentence

> "The patient is **not** allergic to penicillin."

Suppose another sentence is produced by accidentally modifying one word,

> "The patient is allergic to penicillin."

Only one word has changed.

The acoustic energy difference may be extremely small.

However,

the semantic meaning changes completely.

Now consider another modification.

Suppose we slightly alter the pronunciation of

> "patient"

without affecting intelligibility.

The acoustic difference may actually be larger,

yet the semantic meaning remains essentially unchanged.

This example demonstrates an important observation.

Large acoustic changes do not necessarily imply large semantic changes.

Similarly,

small acoustic changes may produce enormous semantic changes.

Therefore,

semantic importance cannot be determined solely from low-level signal properties.

---

## Observation 1.1

Semantic importance is **not proportional** to acoustic energy.

Likewise,

semantic importance is **not proportional** to loudness,

duration,

or frequency.

Instead,

semantic importance depends upon contextual meaning.

---

> [!IMPORTANT]
> This observation explains why traditional psychoacoustic watermarking methods cannot explicitly preserve semantic information. They optimize perceptual quality rather than semantic integrity.

---

# 1.5 True Semantic Importance and Estimated Semantic Importance

Since semantic importance cannot be observed directly,

we distinguish between two different quantities.

The first quantity represents the **true**, but unknown, semantic importance.

The second represents the value estimated by our mathematical model.

Making this distinction is essential because confusing these quantities leads to incorrect scientific reasoning.

---

## Definition 1.2 — True Semantic Importance

Let

\[
T_i
\]

denote the true semantic importance of the

\[
i^{\text{th}}
\]

embedding.

Mathematically,

\[
T_i\in\mathbb{R}.
\]

The collection of all true semantic importance values is

\[
\mathcal{T}
=
\{T_1,T_2,\ldots,T_N\}.
\]

The crucial point is that

\[
T_i
\]

is **unknown**.

There exists no algorithm capable of measuring

\[
T_i
\]

directly.

Therefore,

throughout this notebook,

\[
T_i
\]

is treated as an unobservable latent variable.

---

## Definition 1.3 — Estimated Semantic Importance

Instead of computing

\[
T_i,
\]

our framework constructs an estimator,

\[
\hat{S}_i,
\]

where

\[
\hat{S}_i
=
f(E_i).
\]

Here,

- \(E_i\) denotes the observed embedding,
- \(f(\cdot)\) denotes the estimator,
- \(\hat{S}_i\) denotes the estimated semantic importance.

The symbol

\[
\hat{\;}
\]

(read as "hat")

indicates that the quantity is an estimate rather than the unknown truth.

---

## Relationship Between the Two

Ideally,

our estimator should satisfy

\[
\hat{S}_i
\approx
T_i.
\]

Notice that we deliberately write

\[
\approx
\]

rather than

\[
=
\]

because equality cannot generally be guaranteed.

The objective of SemBind-Audio is therefore **not** to recover the exact semantic importance.

Instead,

our objective is to estimate it as accurately as possible.

---

> [!FACT]
> Every machine learning prediction is fundamentally an estimate of an unknown quantity. The estimated value should be viewed as an approximation rather than absolute truth.

---

# 1.6 Why Do We Need an Estimator?

Suppose an ideal oracle existed that could instantly provide

\[
T_i
\]

for every embedding.

The watermarking problem would become trivial.

We would simply

1. compute every

\[
T_i,
\]

2. rank all embeddings,

3. embed the watermark into the least important regions.

Unfortunately,

no such oracle exists.

Consequently,

we require a mathematical estimator.

The estimator serves as a bridge between

observable quantities

and

latent semantic importance.

Instead of measuring semantic importance,

the estimator infers it from information contained within the embeddings.

---

## General Estimation Framework

Let

\[
\mathcal{O}
\]

denote the collection of observable quantities.

Initially,

we leave this set unspecified.

Later chapters will determine exactly which observable variables should be included.

Possible candidates include

- cosine similarity,
- contextual consistency,
- embedding density,
- neighborhood structure,
- statistical dispersion.

The estimator therefore has the general form

\[
\hat{S}_i
=
f(\mathcal{O}_i),
\]

where

\[
\mathcal{O}_i
\]

represents the observable evidence associated with the

\[
i^{\text{th}}
\]

embedding.

Notice that this formulation is intentionally general.

At this stage,

we do **not** assume which observable variables are most informative.

That question forms one of the primary research problems addressed in subsequent chapters.

---

> [!RESEARCH NOTE]
> One of the novel aspects of SemBind-Audio is identifying which observable embedding properties provide the most reliable estimate of semantic importance. Existing watermarking methods rarely formulate this estimation problem explicitly.

---

# 1.7 Mathematical Objective

We may now express the overall research objective more formally.

---

## Given

A sequence of contextual embeddings

\[
\mathcal{E}
=
\{E_1,E_2,\ldots,E_N\},
\]

where

\[
E_i
\in
\mathbb{R}^{768}.
\]

---

## Unknown

The latent semantic importance values

\[
\mathcal{T}
=
\{T_1,T_2,\ldots,T_N\}.
\]

---

## Required

Construct an estimator

\[
f
\]

such that

\[
\hat{S}_i
=
f(E_i)
\]

provides the most reliable approximation of

\[
T_i.
\]

---

## Optimization Goal

Conceptually,

the desired estimator minimizes the estimation error,

\[
\left|
T_i
-
\hat{S}_i
\right|.
\]

Because

\[
T_i
\]

is unknown,

this error cannot be computed directly.

Instead,

the quality of the estimator must be evaluated indirectly through carefully designed experiments.

This distinction is extremely important.

It prevents us from making unsupported claims regarding estimator accuracy.

---

> [!IMPORTANT]
> Since the ground-truth semantic importance is unavailable, estimator validation must rely on indirect empirical evidence rather than direct numerical error measurement.

---

# 1.8 Assumptions

Every mathematical model relies upon assumptions.

Explicitly stating these assumptions improves scientific clarity and allows future researchers to understand the scope and limitations of the proposed framework.

---

## Assumption A1

HuBERT embeddings preserve sufficient contextual information to support downstream semantic analysis.

---

## Assumption A2

Semantic importance varies across different speech segments.

In other words,

not every embedding contributes equally to the overall meaning of the audio.

---

## Assumption A3

Semantic importance is correlated with one or more observable properties of the embedding space.

Otherwise,

estimation would be impossible.

---

## Assumption A4

The embedding process itself does not perfectly encode semantic importance into any single vector component.

Instead,

semantic information is distributed across the representation.

---

## Assumption A5

An estimator constructed from multiple observable quantities is expected to outperform an estimator based on a single observable quantity.

This assumption motivates the hybrid scoring framework developed in later chapters.

---

## Assumption A6

Embedding watermark information into regions estimated to have lower semantic importance will preserve semantic integrity better than selecting regions solely according to low-level acoustic criteria.

This assumption constitutes the principal hypothesis of the SemBind-Audio framework and will be evaluated experimentally.

---

# 1.9 Scope of the Research

A good scientific formulation clearly defines not only what it intends to solve but also what lies outside its scope. This prevents ambiguity, unrealistic expectations, and incorrect interpretation of the proposed methodology.

The SemBind-Audio framework focuses specifically on **semantic-aware watermark region selection**.

More precisely, the research addresses the following problem:

> Given a sequence of contextual speech embeddings, determine which regions are most suitable for watermark embedding based on their estimated semantic importance.

The framework therefore concentrates on the **decision-making process preceding watermark insertion**, rather than on the watermark embedding algorithm itself.

---

## In Scope

The proposed formulation includes the following research components.

### 1. Semantic Representation

The framework assumes that a pretrained speech foundation model (HuBERT) produces contextual embeddings that serve as the primary representation of the speech signal.

---

### 2. Semantic Importance Estimation

The framework develops a mathematical estimator capable of assigning a semantic importance score to each embedding.

This constitutes the central contribution of the research.

---

### 3. Embedding Region Selection

The estimated semantic importance scores are used to determine which embedding regions are more suitable for watermark insertion.

---

### 4. Semantic Preservation

The framework aims to preserve the semantic integrity of speech by avoiding modification of highly informative regions whenever possible.

---

### 5. Experimental Validation

The proposed estimator is evaluated experimentally using objective metrics, subjective analysis, robustness testing, and comparison with existing watermarking strategies.

---

## Out of Scope

The following problems are intentionally **not** addressed by this research.

### Speech Recognition

SemBind-Audio is not intended to improve automatic speech recognition performance.

---

### Speech Synthesis

The framework does not generate speech.

---

### Speaker Identification

Speaker recognition is not the primary objective.

---

### Emotion Recognition

Although embeddings may contain emotional information, emotion estimation is outside the scope of this work.

---

### HuBERT Training

The HuBERT model is used as a pretrained feature extractor.

No retraining or architectural modification of HuBERT is proposed.

---

### General Representation Learning

This work does not introduce a new representation learning algorithm.

Instead, it builds upon existing pretrained representations.

---

> [!IMPORTANT]
> The novelty of SemBind-Audio lies **after** representation extraction. The research contribution begins once contextual embeddings have been obtained.

---

# 1.10 What This Chapter Does NOT Claim

Scientific integrity requires distinguishing between hypotheses, assumptions, and experimentally verified facts.

Accordingly, this chapter deliberately avoids making claims that cannot yet be supported by evidence.

The following statements are **not** claimed in this chapter.

---

## We Do Not Claim That HuBERT Encodes Perfect Semantics

HuBERT produces powerful contextual representations.

However,

these representations are approximations learned from self-supervised training.

They should not be interpreted as perfect semantic descriptions.

---

## We Do Not Claim That Semantic Importance Can Be Measured Exactly

Semantic importance is modeled as a latent variable.

Consequently,

the framework estimates semantic importance rather than measuring it directly.

---

## We Do Not Claim That One Observable Quantity Is Sufficient

At this stage,

we do not assume that cosine similarity,

vector norm,

context,

density,

or any other individual quantity alone is sufficient to estimate semantic importance.

The suitability of these candidates will be investigated systematically in later chapters.

---

## We Do Not Claim Superior Performance Without Evidence

Although the proposed formulation is motivated by sound theoretical reasoning,

its effectiveness must ultimately be demonstrated experimentally.

Until such experiments are completed,

performance improvements remain hypotheses rather than established facts.

---

> [!SCIENTIFIC PRINCIPLE]
> A research hypothesis should never be presented as an established fact before experimental validation.

---

# 1.11 Research Contributions of This Formulation

Having established the problem mathematically, we now summarize the conceptual contributions introduced in this chapter.

---

## Contribution C1 — Reformulating Watermark Region Selection

Traditional audio watermarking methods generally formulate region selection as an acoustic optimization problem.

SemBind-Audio reformulates the problem as one of **semantic importance estimation**.

This change shifts the focus from low-level signal properties to high-level learned representations.

---

## Contribution C2 — Treating Semantic Importance as a Latent Variable

Instead of assuming semantic importance can be directly computed,

the framework explicitly models it as an unknown latent variable.

This provides a statistically consistent foundation for subsequent estimation.

---

## Contribution C3 — Estimation-Based Framework

Rather than relying on handcrafted heuristics,

SemBind-Audio introduces the concept of estimating semantic importance from observable embedding properties.

This estimation perspective forms the mathematical core of the framework.

---

## Contribution C4 — Separation Between Truth and Estimation

The framework distinguishes between

\[
T_i
\]

(the unknown true semantic importance)

and

\[
\hat{S}_i
\]

(the estimated semantic importance).

Although this distinction appears simple,

it is frequently omitted in existing watermarking literature.

Making this separation explicit improves theoretical clarity.

---

## Contribution C5 — Foundation for Hybrid Semantic Scoring

The current chapter intentionally avoids selecting a specific estimator.

Instead,

it establishes the mathematical foundation upon which later chapters will investigate multiple observable indicators and construct a hybrid semantic scoring framework.

---

> [!OUR CONTRIBUTION]
> The principal contribution of this chapter is **problem formulation**, not algorithm design. The mathematical estimator itself will be derived in subsequent chapters.

---

# Notation Introduced in This Chapter

| Symbol | Meaning |
|---------|---------|
| \(X\) | Input speech waveform |
| \(\mathcal{E}\) | Set of all embeddings |
| \(E_i\) | Embedding corresponding to the \(i^{\text{th}}\) speech segment |
| \(N\) | Number of embeddings |
| \(T_i\) | Unknown true semantic importance |
| \(\hat{S}_i\) | Estimated semantic importance |
| \(f(\cdot)\) | Semantic importance estimator |
| \(\mathcal{O}\) | Observable quantities used for estimation |

---

# Chapter Summary

This chapter formally defined the central research problem addressed by SemBind-Audio.

Rather than embedding watermark information according to acoustic characteristics alone, the proposed framework seeks to estimate the semantic importance of contextual speech embeddings.

To achieve this, semantic importance was modeled as an unobservable latent variable.

The distinction between the unknown true semantic importance,

\[
T_i,
\]

and the estimated semantic importance,

\[
\hat{S}_i,
\]

was introduced to provide a mathematically rigorous foundation for the framework.

The chapter also established the assumptions, scope, and scientific boundaries of the proposed research.

Most importantly,

it transformed the watermark region selection problem into an estimation problem,

laying the theoretical foundation for all subsequent chapters.

---

# Key Takeaways

1. Semantic importance cannot be measured directly.

2. Semantic importance is modeled as a latent variable.

3. Watermark region selection becomes an estimation problem.

4. Observable embedding properties must be used to infer semantic importance.

5. The quality of the estimator must be validated experimentally.

6. The mathematical estimator has not yet been defined.

7. Constructing that estimator constitutes the next stage of the research.

---

# Reviewer Questions

1. Why is semantic importance modeled as a latent variable?

2. What distinguishes an observable variable from a latent variable?

3. Why is estimating semantic importance more appropriate than attempting to measure it directly?

4. Why does SemBind-Audio distinguish between \(T_i\) and \(\hat{S}_i\)?

5. What assumptions are required before semantic importance estimation becomes possible?

6. Which parts of the watermarking pipeline are inside the scope of this research?

7. Why should hypotheses not be presented as established facts?

8. Why does this chapter stop before defining the estimator?

9. What is the principal theoretical contribution of this chapter?

10. How does this formulation differ conceptually from traditional psychoacoustic watermarking?

---

# References

## [R1] Bishop, C. M. *Pattern Recognition and Machine Learning.*

**Purpose**

Provides the statistical foundations for latent variables, estimation, and probabilistic modeling.

**Used for**

- Latent variable formulation
- Estimation theory
- Observable vs. hidden variables

**Not used for**

- Audio watermarking
- Transformer embeddings

---

## [R2] Goodfellow, I., Bengio, Y., & Courville, A. *Deep Learning.*

**Purpose**

Provides the theoretical basis for representation learning and learning unknown functions from data.

**Used for**

- Representation learning
- Function approximation
- Deep learning foundations

**Not used for**

- Semantic-aware watermarking

---

## [R3] Hsu, W.-N., et al. (2021). *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units.*

**Purpose**

Supports the use of contextual speech embeddings as learned representations.

**Used for**

- HuBERT embeddings
- Contextual speech representations

**Not used for**

- Semantic importance estimation
- Watermark region selection

---

# Bridge to Chapter 02

At this point,

the problem has been formulated,

but the estimator

\[
f(\cdot)
\]

remains completely undefined.

The next logical question is therefore:

> **Which observable properties of an embedding are actually informative enough to estimate semantic importance?**

To answer this question, the next chapter investigates a series of candidate semantic indicators—including cosine similarity, contextual consistency, vector norms, neighborhood density, entropy, and other embedding-space properties—before selecting the most theoretically justified components for the SemBind-Audio semantic importance estimator.

This marks the transition from **problem formulation** to **estimator design**, where the original mathematical contribution of SemBind-Audio begins.

# SemBind-Audio Research Mathematics Notebook

# Chapter 02

# Candidate Semantic Indicators

---

# Learning Objectives

After completing this chapter, the reader should be able to:

- Understand why semantic importance cannot be observed directly.
- Identify observable properties of embedding spaces.
- Analyze the strengths and weaknesses of different semantic indicators.
- Understand why no single observable quantity is sufficient.
- Appreciate the need for a hybrid semantic importance estimator.
- Distinguish between evidence and semantic importance itself.

---

# Prerequisites

The reader should understand

- embeddings,
- latent spaces,
- representation learning,
- observable variables,
- latent variables,
- semantic importance formulation,

introduced in Chapters 00 and 01.

---

# Chapter Overview

In Chapter 01 we formulated semantic importance as an unknown latent variable,

\[
T_i.
\]

Since

\[
T_i
\]

cannot be measured directly,

our objective is to estimate it using observable properties of the embedding space.

However,

before constructing an estimator,

an important scientific question must be answered.

> **Which observable quantities actually contain evidence about semantic importance?**

This chapter investigates several possible candidates.

Rather than immediately proposing a mathematical formula,

we first analyze the theoretical strengths and weaknesses of each candidate.

This systematic analysis is essential because scientific models should be derived from evidence rather than intuition.

The outcome of this chapter is not a final estimator.

Instead,

it establishes the evidence required to justify the estimator developed in subsequent chapters.

---

# 2.1 Observable Evidence

Suppose

\[
E_i
\]

denotes the embedding of the

\[
i^{\text{th}}
\]

speech segment.

Recall from Chapter 01 that

semantic importance,

\[
T_i,
\]

cannot be observed directly.

Therefore,

our estimator must rely upon measurable properties derived from

\[
E_i.
\]

We call such measurable properties **observable evidence**.

---

## Definition 2.1 — Observable Evidence

Observable evidence is any numerical quantity that can be computed directly from embeddings and that may contain information useful for estimating semantic importance.

Observable evidence differs fundamentally from semantic importance.

Semantic importance is unknown.

Observable evidence is measurable.

Consequently,

observable evidence serves as indirect information from which semantic importance may be inferred.

---

## Candidate Observable Quantities

Several measurable quantities naturally arise from embedding spaces.

Examples include

- cosine similarity,
- Euclidean distance,
- vector norm,
- contextual consistency,
- neighborhood density,
- embedding entropy,
- attention scores,
- statistical dispersion,
- local embedding structure.

Each of these quantities captures a different aspect of the embedding geometry.

The central question becomes

> Which of these quantities provides meaningful evidence about semantic importance?

---

> [!IMPORTANT]
> Observable evidence is **not** semantic importance. It is only information that may help estimate semantic importance.

---

# 2.2 Criteria for a Good Semantic Indicator

Before evaluating individual candidates,

we first establish the characteristics that any useful semantic indicator should satisfy.

Without such criteria,

there is no principled way to compare competing indicators.

Accordingly,

we define the following requirements.

---

## Criterion C1 — Semantic Sensitivity

The indicator should respond to meaningful semantic changes.

If two speech segments differ substantially in semantic content,

the indicator should reflect that difference.

---

## Criterion C2 — Context Awareness

Speech meaning depends upon context.

Therefore,

a useful indicator should incorporate contextual relationships rather than treating every segment independently.

---

## Criterion C3 — Robustness

Minor acoustic variations,

background noise,

or recording artifacts

should not dramatically alter the indicator.

Otherwise,

the estimator becomes unstable.

---

## Criterion C4 — Computational Efficiency

The indicator should be computable efficiently,

particularly for long audio recordings.

An estimator with prohibitive computational complexity may be impractical despite strong theoretical properties.

---

## Criterion C5 — Generality

The indicator should not rely upon assumptions specific to a single dataset,

speaker,

language,

or recording condition.

Instead,

it should generalize across diverse speech inputs.

---

## Criterion C6 — Compatibility

The indicator should integrate naturally with other observable quantities.

This requirement becomes important if multiple indicators are ultimately combined into a hybrid estimator.

---

> [!FACT]
> These criteria evaluate the usefulness of an indicator—not its correctness. Since semantic importance is unobservable, no indicator can be declared universally correct without empirical validation.

---

# 2.3 Candidate 1 — Cosine Similarity

Among all observable quantities,

cosine similarity is perhaps the most widely used measure in representation learning.

Instead of comparing vector magnitudes,

cosine similarity measures the angle between two vectors.

Consequently,

it emphasizes directional similarity rather than absolute scale.

Because embeddings primarily encode information through their geometric orientation,

cosine similarity has become the standard measure for comparing learned representations.

For this reason,

cosine similarity is the first candidate investigated in SemBind-Audio.

---

## Intuition

Suppose two embeddings point in nearly the same direction.

Their corresponding speech segments are likely to contain related information.

Conversely,

embeddings pointing in very different directions may correspond to different contextual meanings.

Therefore,

cosine similarity appears to contain useful evidence regarding semantic relationships.

---

## Potential Advantages

Cosine similarity possesses several desirable properties.

### Advantage A1

Insensitive to overall vector magnitude.

---

### Advantage A2

Widely validated across representation learning literature.

---

### Advantage A3

Computationally efficient.

---

### Advantage A4

Naturally compatible with transformer embedding spaces.

---

### Advantage A5

Provides continuous similarity values,

making ranking straightforward.

---

## Initial Hypothesis

One possible hypothesis is

> Embeddings that are highly similar to the overall semantic content of the audio may carry greater semantic importance.

Although intuitively appealing,

this hypothesis requires careful examination.

---

# 2.4 Is Cosine Similarity Alone Sufficient?

The previous section identified cosine similarity as a promising observable quantity.

However, scientific research does not stop after finding a plausible candidate.

A more important question must now be answered.

> **Can cosine similarity alone estimate semantic importance?**

If the answer is yes,

there is no need to investigate additional indicators.

If the answer is no,

we must determine why it fails and identify the missing information.

This section therefore performs a critical analysis of cosine similarity.

---

## What Does Cosine Similarity Actually Measure?

Suppose we have two embeddings

\[
E_i,E_j\in\mathbb{R}^{768}.
\]

Their cosine similarity is

\[
\text{Cos}(E_i,E_j)
=
\frac{E_i\cdot E_j}
{\|E_i\|\|E_j\|}.
\]

Notice an important observation.

The equation contains only

- vector components,
- dot product,
- vector magnitudes.

There is **no explicit semantic variable** anywhere in the equation.

Therefore,

cosine similarity does **not** directly measure semantic importance.

Instead,

it measures **geometric alignment** between two vectors.

Semantic interpretation arises only because transformer models organize semantically related representations into nearby regions of the embedding space.

Thus,

cosine similarity provides indirect evidence rather than direct measurement.

---

> [!IMPORTANT]
> Cosine similarity measures geometric similarity—not semantic importance. Any relationship between the two is indirect and must be justified empirically.

---

# 2.5 Counterexample 1 — The "Not" Problem

One of the most instructive examples comes from natural language semantics.

Consider the sentence

> "The patient is allergic to penicillin."

Now insert one word.

> "The patient is **not** allergic to penicillin."

Only one word has changed.

From an acoustic perspective,

the additional word is short and occupies very little time.

However,

its semantic contribution is enormous.

Removing or modifying this single word completely reverses the meaning of the sentence.

Suppose the embedding corresponding to the word

> "not"

is compared with a global representation of the sentence.

Its cosine similarity may not necessarily be the largest among all embeddings.

Nevertheless,

its semantic importance is exceptionally high.

This observation immediately reveals an important limitation.

High semantic importance does not necessarily imply high cosine similarity.

---

## Observation 2.1

Cosine similarity cannot distinguish between

- semantic centrality,

and

- semantic necessity.

A word may be essential for preserving meaning while exhibiting only moderate geometric similarity.

---

# 2.6 Counterexample 2 — Frequently Repeated Words

Consider another sentence.

> "Please, please, please help me."

The word

> "please"

appears multiple times.

Because the contextual embeddings are likely to be similar,

their cosine similarities with one another may be very high.

Does this imply that every occurrence possesses equal semantic importance?

Not necessarily.

Removing one repeated occurrence often has minimal influence on the overall meaning.

Therefore,

large cosine similarity alone cannot determine semantic necessity.

---

# 2.7 Counterexample 3 — Silence

Suppose a short pause occurs between two words.

The pause may produce embeddings that remain relatively consistent with surrounding speech.

Consequently,

their cosine similarities with neighboring embeddings could remain reasonably high.

Yet,

embedding watermark information within silence is unlikely to alter semantic meaning.

This example demonstrates another limitation.

Semantic importance depends not only on similarity,

but also on the linguistic role of the corresponding speech segment.

---

# 2.8 Conclusion on Cosine Similarity

The previous examples demonstrate an important conclusion.

Cosine similarity certainly contains useful semantic evidence.

However,

it is insufficient by itself.

It captures one aspect of embedding geometry,

namely,

directional similarity.

It does **not** capture

- contextual necessity,
- linguistic function,
- semantic dependency,
- discourse importance.

Therefore,

cosine similarity should be regarded as one component of a larger estimator rather than a complete solution.

---

> [!RESEARCH OBSERVATION]
> Cosine similarity satisfies Criteria C1, C3, C4, and C6 reasonably well, but does not fully satisfy Criterion C2 because it lacks an explicit mechanism for modeling contextual importance.

---

# 2.9 Candidate 2 — Contextual Consistency

Human language is inherently contextual.

The meaning of an individual word depends heavily upon surrounding words.

Similarly,

the meaning of one speech segment depends upon neighboring speech segments.

Consequently,

another possible indicator is **contextual consistency**.

Rather than comparing one embedding with another,

we compare an embedding with its local neighborhood.

Intuitively,

if removing or modifying a segment substantially disrupts the contextual flow,

that segment may possess greater semantic importance.

---

## Why Context Appears Promising

Transformer models—including HuBERT—already encode contextual information through self-attention.

Therefore,

neighboring embeddings are not independent.

Instead,

they influence one another during representation learning.

This suggests that contextual consistency may provide evidence unavailable through pairwise cosine similarity alone.

---

## Potential Advantages

Contextual consistency offers several strengths.

### Advantage A1

Captures local semantic continuity.

---

### Advantage A2

Reflects relationships among neighboring speech segments.

---

### Advantage A3

Naturally complements cosine similarity.

---

### Advantage A4

Compatible with transformer-based contextual embeddings.

---

# 2.10 Is Context Alone Sufficient?

Although contextual consistency appears attractive,

it also exhibits important weaknesses.

Consider the sentence

> "Thank... you."

The pause between the two words contributes to contextual continuity.

However,

its semantic contribution remains negligible.

Similarly,

background breathing,

filled pauses,

or prolonged silence

may exhibit strong contextual continuity while contributing little semantic information.

Therefore,

context alone cannot determine semantic importance.

---

## Observation 2.2

Contextual consistency measures how smoothly a segment fits within its neighborhood.

It does **not** necessarily measure how important that segment is to the overall semantic meaning.

Consequently,

contextual consistency should also be regarded as partial evidence rather than a complete estimator.

---

# 2.11 Interim Conclusion

We have now examined two candidate indicators.

| Candidate | Useful? | Sufficient? | Reason |
|-----------|----------|-------------|--------|
| Cosine Similarity | ✔ Yes | ✘ No | Captures geometric similarity but not semantic necessity |
| Contextual Consistency | ✔ Yes | ✘ No | Captures contextual flow but not overall semantic importance |

The evidence collected so far suggests an important emerging pattern.

Neither indicator individually satisfies all six criteria established earlier.

However,

each captures information absent from the other.

This observation motivates the next stage of our investigation,

where additional observable quantities will be examined before constructing a unified semantic importance estimator.

---

> [!OUR CONTRIBUTION]
> Rather than selecting the first plausible observable quantity, SemBind-Audio systematically evaluates multiple candidate indicators, identifies their individual limitations, and argues that semantic importance should be inferred from complementary evidence rather than any single geometric property.


