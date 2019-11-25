using LinearAlgebra: diag, Diagonal, svd

@doc doc"""
    FixedRankMatrices{M,N,K} <: Manifold

The manifold of $m\times n$ real-valued matrices of fixed rank $k$, i.e.
````math
\mathcal M = \{ X \in \mathbb R^{m\times n\} : \operatorname{rank}(X) = k \}
````
A tangent space $T_x\mathcal M$ at a point $x\in\mathcal M$ with $x=USV^\mathrm{T}$
is given by
````math
T_x\mathcal M = \bigl\{ UMV^\mathrm{T} + U_xV^\mathrm{T} + UV_x^\mathrm{T} : 
    M \in \mathbb R^{k\times k},
    U_x \in \mathbb R^{m\times k},
    V_x \in \mathbb R^{n\times k}
    \text{ such that }
    U_x^\mathrm{T}U = 0_k,
    V_x^\mathrm{T}V = 0_k
\bigr\},
````
where $0_k$ is the $k\times k$ zero matrix
For more details, see for example
> Vandereycken, Bart: "Low-rank matrix completion by Riemannian Optimization –
> extended bersion", arXiv: [1209.3834](https://arxiv.org/abs/1209.3834)
"""
struct FixedRankMatrices{M,N,T} <: Manifold end
FixedRankMatrices(m::Int, n::Int, k::Int) = FixedRankMatrices{m,n,k}()
