@doc raw"""
    GeneralizedBurresWassertseinMetric{T<:AbstractMatrix} <: AbstractMetric

The generalized Bures Wasserstein metric for symmetric positive definite matrices, see[^HanMishraJawanpuriaGao2021].

This metric internally stores the symmetric positive definite matrix ``M`` to generalise the metric,
where the name also follows the mentioned preprint.

[^^HanMishraJawanpuriaGao2021]:
    > Han, A., Mishra, B., Jawanpuria, P., Gao, J.:
    > Generalized Bures-Wasserstein geometry for positive definite matrices
    > arXiv: [2110.10464](https://arxiv.org/abs/2110.10464).
"""
struct GeneralizedBuresWassersteinMetric{T<:AbstractMatrix} <: RiemannianMetric
    M::T
end
