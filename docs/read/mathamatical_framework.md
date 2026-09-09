---

# 4.4 The Dot Product

To quantify the geometric relationship between two vectors, we first introduce one of the most fundamental operations in linear algebra: the **dot product**.

Given two vectors

\[
A = [a_1, a_2, \ldots, a_d]^T
\]

and

\[
B = [b_1, b_2, \ldots, b_d]^T,
\]

their dot product is defined as

\[
A \cdot B =
\sum_{k=1}^{d} a_k b_k.
\]

This operation produces a single scalar value representing the interaction between the corresponding components of the two vectors.

From a computational perspective, the dot product is simply the sum of pairwise multiplications. However, its geometric interpretation is considerably more informative.

---

## Geometric Interpretation

Suppose two vectors originate from the same point in the embedding space.

The dot product depends upon two factors:

1. the lengths of the vectors, and
2. the angle between them.

If both vectors point in approximately the same direction, the dot product becomes large and positive.

If the vectors are orthogonal, their dot product becomes zero.

If they point in opposite directions, the dot product becomes negative.

Therefore, the dot product captures both **magnitude** and **direction** simultaneously.

This dual dependence is both a strength and a limitation.

While directional agreement is useful for measuring semantic similarity, dependence on vector magnitude makes direct comparison between embeddings unreliable.

This motivates the normalization process introduced in the next section.

---

> [!IMPORTANT]
> The dot product alone is **not** a similarity measure because its value depends on both vector direction and vector magnitude.

---

# 4.5 From Dot Product to Cosine Similarity

The geometric relationship between two vectors can be expressed using the following identity:

\[
A \cdot B
=
\|A\| \, \|B\| \cos\theta,
\]

where

- \(\|A\|\) is the Euclidean norm of \(A\),
- \(\|B\|\) is the Euclidean norm of \(B\),
- \(\theta\) is the angle between the vectors.

This equation provides a direct connection between algebra and geometry.

The left-hand side is computed numerically using vector components.

The right-hand side explains the same quantity geometrically.

Notice that the dot product increases under two different conditions:

- when the vectors become longer, and
- when the angle between them becomes smaller.

For semantic comparison, only the second property is desirable.

We wish to compare **directional alignment**, not vector length.

Consequently, the influence of vector magnitude must be removed.

---

# 4.6 Normalization

Starting from

\[
A \cdot B
=
\|A\| \, \|B\| \cos\theta,
\]

divide both sides by

\[
\|A\| \, \|B\|.
\]

This yields

\[
\boxed{
\cos\theta
=
\frac{A \cdot B}
{\|A\| \, \|B\|}
}
\]

This quantity is known as **cosine similarity**.

Unlike the dot product, cosine similarity depends only on the angle between two vectors.

The influence of vector magnitude has been completely removed.

Consequently, two vectors pointing in the same direction obtain identical cosine similarity regardless of their lengths.

This property makes cosine similarity particularly well suited for learned embedding spaces, where the orientation of an embedding typically carries more semantic information than its magnitude.

---

## Why Normalization Matters

Consider two vectors,

\[
A=(1,2)
\]

and

\[
B=(2,4).
\]

The second vector is simply a scaled version of the first.

Their dot product is

\[
A \cdot B
=
1\times2 + 2\times4
=
10.
\]

Although this value is positive, it does not directly indicate perfect alignment.

Computing cosine similarity gives

\[
\cos(\theta)
=
1,
\]

indicating that both vectors point in exactly the same direction.

Thus, normalization correctly identifies directional agreement even when magnitudes differ.

---

> [!FACT]
> Cosine similarity is invariant to positive scaling. Multiplying an embedding by a positive constant changes its magnitude but does not change its cosine similarity with another vector.

---

# 4.7 Global Semantic Alignment in SemBind-Audio

The previous derivation applies to any pair of vectors.

Within SemBind-Audio, we substitute

\[
A = E_i,
\]

where

\[
E_i
\]

is the contextual embedding corresponding to the \(i^{\text{th}}\) speech segment,

and

\[
B = G,
\]

where

\[
G
\]

is the Global Audio Representation introduced in Chapter 03.

The cosine similarity between an embedding and the complete recording is therefore

\[
\boxed{
A_i
=
\frac{E_i \cdot G}
{\|E_i\| \, \|G\|}
}
\]

where

\[
A_i
\]

denotes the **Global Semantic Alignment Score** of the \(i^{\text{th}}\) embedding.

This score measures the degree to which an individual speech segment is geometrically aligned with the overall semantic representation of the recording.

A higher value indicates stronger alignment between the local embedding and the global semantic context.

A lower value indicates weaker alignment.

---

## Definition 4.1 — Global Semantic Alignment Score

The **Global Semantic Alignment Score** is defined as the cosine similarity between an individual contextual embedding and the Global Audio Representation.

Mathematically,

\[
A_i
=
\frac{E_i \cdot G}
{\|E_i\| \, \|G\|}.
\]

The score satisfies

\[
-1 \leq A_i \leq 1.
\]

Importantly, \(A_i\) is **not** interpreted as the semantic importance of the speech segment.

Instead, it provides one observable piece of evidence describing how closely the embedding aligns with the semantic content of the complete recording.

---

> [!OUR CONTRIBUTION]
> SemBind-Audio introduces the Global Semantic Alignment Score as one component of a broader semantic importance estimator. Unlike approaches that directly interpret cosine similarity as importance, our framework treats it as a complementary observable quantity that must be integrated with additional semantic evidence.

