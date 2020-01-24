@doc doc"""
    GeneralizedStiefel{n,k,B,T} <: Manifold

The Generalized Stiefel manifold consists of all $n\times k$, $n\geq k$ orthonormal matrices w.r.t. an arbitrary scalar product `B`, i.e.

````math
\mathcal M = \{ x \in \mathbb F^{n\times k} : x^{\mathrm{H}}\cdot B \cdot x = I_k \},
````

where $\mathbb F \in \{\mathbb R, \mathbb C\}$,
$\cdot^{\mathrm{H}}$ denotes the complex conjugate transpose or Hermitian, and
$I_n \in \mathbb R^{n\times n}$ denotes the $k \times k$ identity matrix.


In the case $B=I_k$, one gets the usual [`Stiefel`](@ref) manifold.

# Constructor
    Stiefel(n,k,B=I,F=ℝ)

Generate the (real-valued) Stiefel manifold of $n\times k$ dimensional orthonormal matrices.
"""
struct GeneralizedStiefel{n,k,B,F} <: Manifold end

GeneralizedStiefel(n::Int, k::Int, B = I, F::AbstractNumbers = ℝ) = GeneralizedStiefel{n,k,B,F}()