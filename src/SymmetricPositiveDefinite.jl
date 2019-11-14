using LinearAlgebra: eigen, eigvals, eigvecs, Symmetric, Diagonal, factorize, tr, norm

@doc doc"""
    SymmetricPositiveDefinite{N} <: Manifold

The manifold of symmetric positive definite matrices, i.e.

```math
\mathcal P(n) =
\bigl\{
    x \in \mathbb R^{n\\times n} :
    \xi^\mathrm{T}x\xi > 0 \text{ for all } \xi \in \mathbb R^{n}\backslash\{0\}
\bigr\}
```

# Constructor

    SymmetricPositiveDefinite(n)

generates the $\mathcal P(n) \subset \mathbb R^{n\times n}$
"""
struct SymmetricPositiveDefinite{N} <: Manifold end
SymmetricPositiveDefinite(n::Int) = SymmetricPositiveDefinite{n}()

@doc doc"""
    manifold_dimension(::SymmetricPositiveDefinite{N})

returns the dimension of the manifold [`SymmetricPositiveDefinite`](@ref) $\mathcal P(n)$, N\in \mathbb N$, i.e.
```math
    \frac{n(n+1)}{2}    
```
"""
@generated manifold_dimension(::SymmetricPositiveDefinite{N}) where {N} = div(N*(N+1), 2)

@doc doc"""
    LinearAffineMetric <: Metric

The linear affine metric is the metric for symmetric positive definite matrices, that employs
matrix logarithms and exponentials, which yields a linear and affine metric.
"""
struct LinearAffineMetric <: Metric end

@doc doc"""
    LogEuclideanMetric <: Metric

The LogEuclidean Metric consists of the Euclidean metric applied to all elements after mapping them
into the Lie Algebra, i.e. performing a matrix logarithm beforehand.
"""
struct LogEuclideanMetric <: Metric end

@doc doc"""
    distance(P,x,y)

computes the distance on the [`SymmetricPositiveDefinite`](@ref) manifold between `x` and `y`,
which defaults to the [`LinearAffineMetric`](@ref) induces distance.
"""
distance(P::SymmetricPositiveDefinite{N},x,y) where N = distance(MetricManifold(P,LinearAffineMetric()),x,y)

@doc doc"""
    distance(P,x,y)

computes the distance on the [`SymmetricPositiveDefinite`](@ref) manifold between `x` and `y`,
as a [`MetricManifold`](@ref) with [`LinearAffineMetric`](@ref). The formula reads

```math
d_{\mathcal P(n)}(x,y) = \lVert \operatorname{Log}(x^{-\frac{1}{2}}yx^{-\frac{1}{2}})\rVert_{\mathrm{F}}.,
```
where $\operatorname{Log}$ denotes the matrix logarithm and $\lVert\cdot\rVert_{\mathrm{F}}$ denotes the
matrix Frobenius norm.
"""
function distance(P::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric},x,y) where N
    s = real.( eigvals( x,y ) )
    return any(s .<= eps() ) ? 0 : sqrt(  sum( abs.(log.(s)).^2 )  )
end

@doc doc"""
    distance(P,x,y)

computes the distance on the [`SymmetricPositiveDefinite](@ref) manifold between
`x` and `y` as a [`MetricManifold`](@ref) with [`LogEuclideanMetric`](@ref).
The formula reads

```math
    d_{\mathcal P(n)}(x,y) = \lVert \Log x - \Log y \rVert_{\mathrm{F}}
```
where $\operatorname{Log}$ denotes the matrix logarithm and $\lVert\cdot\rVert_{\mathrm{F}}$ denotes the
matrix Frobenius norm.
"""
function distance(P::MetricManifold{SymmetricPositiveDefinite{N},LogEuclideanMetric},x,y) where N
    eX = eigen(Symmetric(x))
    UX = eX.vectors
    SX = eX.values
    eY = eigen(Symmetric(y))
    UY = eY.vectors
    SY = eY.values
    return norm( UX*Diagonal(log.(SX))*transpose(UX) - UY*Diagonal(log.(SY))*transpose(UY))
end

@doc doc"""
    inner(P,x,v,w)

compute the inner product of `v`, `w` in the tangent space of `x` on the [`SymmetricPositiveDefinite`](@ref)
manifold `P`, which defaults to the [`LinearAffineMetric`](@ref).
"""
inner(P::SymmetricPositiveDefinite{N}, x, w, v) where N = inner(MetricManifold(P,LinearAffineMetric()),x,w,v)

@doc doc"""
    inner(P,x,v,w)

compute the inner product of `v`, `w` in the tangent space of `x` on the [`SymmetricPositiveDefinite`](@ref)
manifold `P`, as a [`MetricManifold`](@ref) with [`LinearAffineMetric`](@ref). The formula reads

```math
( v, w)_x = \operatorname{tr}(x^{-1}\xi x^{-1}\nu ),
```
"""
function inner(P::MetricManifold{SymmetricPositiveDefinite{N}, LinearAffineMetric}, x, w, v) where N
    F = factorize(x)
    return tr( ( Symmetric(w) / F ) * ( Symmetric(v) / F ) )
end

@doc doc"""
    exp!(P,y,x,v)

compute the exponential map from `x` with tangent vector `v` on the [`SymmetricPositiveDefinite`](@ref)
manifold with its default metric, [`LinearAffineMetric`](@ref) and modify `y`.
"""
exp!(P::SymmetricPositiveDefinite{N},y,x,v) where N = exp!(MetricManifold(P,LinearAffineMetric()),y,x,v)

@doc doc"""
    exp!(P,y,x,v)

compute the exponential map from `x` with tangent vector `v` on the [`SymmetricPositiveDefinite`](@ref)
as a [`MetricManifold`](@ref) with [`LinearAffineMetric`](@ref) and modify `y`. The formula reads

```math
    \exp_x v = x^{\frac{1}{2}}\operatorname{Exp}(x^{-\frac{1}{2}} v x^{-\frac{1}{2}})x^{\frac{1}{2}},
```
where $\operatorname{Exp}$ denotes to the matrix exponential.
"""
function exp!(P::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric}, y, x, v) where N
    e = eigen(Symmetric(x))
    U = e.vectors
    S = e.values
    Ssqrt = Diagonal( sqrt.(S) )
    SsqrtInv = Diagonal( 1 ./ sqrt.(S) )
    xSqrt = Symmetric(U*Ssqrt*transpose(U))
    xSqrtInv = Symmetric(U*SsqrtInv*transpose(U))
    T = Symmetric(xSqrtInv * v * xSqrtInv)
    eig1 = eigen( T ) # numerical stabilization
    Se = Diagonal( exp.(eig1.values) )
    Ue = eig1.vectors
    xue = xSqrt*Ue
    copyto!(y, xue*Se*transpose(xue) )
    return y
end

@doc doc"""
    log!(P,v,x,y)

compute the logarithmic map at `x` to `y` on the [`SymmetricPositiveDefinite`](@ref)
manifold with its default metric, [`LinearAffineMetric`](@ref) and modify `v`.
"""
log!(P::SymmetricPositiveDefinite{N}, v, x, y) where N = log!(MetricManifold(P,LinearAffineMetric()),v, x, y)

@doc doc"""
    log!(P,v,x,y)

compute the exponential map from `x` to `y` on the [`SymmetricPositiveDefinite`](@ref)
as a [`MetricManifold`](@ref) with [`LinearAffineMetric`](@ref) and modify `v`. The formula reads

```math
\log_x y = x^{\frac{1}{2}}\operatorname{Log}(x^{-\frac{1}{2}} y x^{-\frac{1}{2}})x^{\frac{1}{2}},
```
where $\operatorname{Log}$ denotes to the matrix logarithm.
"""
function log!(P::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric}, v, x, y) where N
    e = eigen(Symmetric(x))
    U = e.vectors
    S = e.values
    Ssqrt = Symmetric( Matrix( Diagonal( sqrt.(S) ) ) )
    SsqrtInv = Symmetric( Matrix( Diagonal( 1 ./ sqrt.(S) ) ) )
    xSqrt = Symmetric( U*Ssqrt*transpose(U) )
    xSqrtInv = Symmetric( U*SsqrtInv*transpose(U) )
    T = Symmetric( xSqrtInv * y * xSqrtInv )
    e2 = eigen( T )
    Se = Matrix( Diagonal( log.(max.(e2.values,eps()) ) ) )
    Ue = e2.vectors
    xue = xSqrt*Ue
    copyto!(v, Symmetric(xue*Se*transpose(xue)))
    return v
end

@doc doc"""
    representation_size(M)

returns the size of an array representing an element on the
[`SymmetricPositiveDefinite`](@ref) manifold `M`,
i.e. $n\times n$, the size of such a symmetric positive definite matrix on
$\mathcal M = \mathcal P(n)$.
"""
function representation_size(::SymmetricPositiveDefinite{N}) where N
    return (N,N)
end

@doc doc"""
    vector_transport_to(P,vto,x,v,y,m::AbstractVectorTransportMethod=ParallelTransport())

compute the vector transport on the [`SymmetricPositiveDefinite`](@ref) with its
default metric, [`LinearAffineMetric`](@ref) and method `m`, which defaults to [`ParallelTransport`](@ref).
"""
vector_transport_to!(P::SymmetricPositiveDefinite{N},vto, x, v, y, m::AbstractVectorTransportMethod) where N = vector_transport_to!(MetricManifold(P,LinearAffineMetric()),vto, x, v, y, m)

@doc doc"""
    vector_transport_to!(P,vto,x,v,y,::ParallelTransport)

compute the parallel transport on the [`SymmetricPositiveDefinite`](@ref) as a
[`MetricManifold`](@ref) with the [`LinearAffineMetric`](@ref).
The formula reads

```math
P_{x\to y}(v) = x^{\frac{1}{2}}
\operatorname{Exp}\bigl(
\frac{1}{2}x^{-\frac{1}{2}}\log_x(y)x^{-\frac{1}{2}}
\bigr)
x^{-\frac{1}{2}}v x^{-\frac{1}{2}}
\operatorname{Exp}\bigl(
\frac{1}{2}x^{-\frac{1}{2}}\log_x(y)x^{-\frac{1}{2}}
\bigr)
x^{\frac{1}{2}},
```

where $\operatorname{Exp}$ denotes the matrix exponential
and [`log`](@ref) the logarithmic map.
"""
function vector_transport_to!(::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric}, vto, x, v, y, ::ParallelTransport) where N
    if norm(x-y)<1e-13
        vto = v
        return vto
    end
    e = eigen(Symmetric(x))
    U = e.vectors
    S = e.values
    Ssqrt = sqrt.(S)
    SsqrtInv = Diagonal( 1 ./ Ssqrt )
    Ssqrt = Diagonal( Ssqrt )
    xSqrt = Symmetric(U*Ssqrt*transpose(U))
    xSqrtInv = Symmetric(U*SsqrtInv*transpose(U))
    tv = Symmetric(xSqrtInv * v * xSqrtInv)
    ty = Symmetric(xSqrtInv * y * xSqrtInv)
    e2 = eigen( ty )
    Se = Diagonal( log.(e2.values) )
    Ue = e2.vectors
    ty2 = Symmetric(Ue*Se*transpose(Ue))
    e3 = eigen( ty2 )
    Sf = Diagonal( exp.(e3.values) )
    Uf = e3.vectors
    xue = xSqrt*Uf*Sf*transpose(Uf)
    vtp = xue * ( 0.5*(tv + transpose(tv)) ) * transpose(xue)
    copyto!(vto, vtp) # symmetrize
    return vto
end

@doc doc"""
    [Ξ,κ] = tangent_orthonormal_basis(M,x,v)

returns a orthonormal basis `Ξ` in the tangent space of `x` on the
[`SymmetricPositiveDefinite`](@ref) manifold `M` with the defrault metric, the
[`LinearAffineMetric`](ref) that diagonalizes the curvature tensor $R(u,v)w$
with eigenvalues `κ` and where the direction `v` has curvature `0`.
"""
tangent_orthonormal_basis(P::SymmetricPositiveDefinite{n},x,v) where n = tangent_orthonormal_basis(MetricManifold(P,LinearAffineMetric()),x,v)

@doc doc"""
    [Ξ,κ] = tangent_orthonormal_basis(M,x,v)

returns a orthonormal basis `Ξ` as a vector of tangent vectors (of length
[`manifold_dimension`](@ref) of `M`) in the tangent space of `x` on the
[`MetricManifold`](@ref of [`SymmetricPositiveDefinite`](@ref) manifold `M` with
[`LinearAffineMetric`](ref) that diagonalizes the curvature tensor $R(u,v)w$
with eigenvalues `κ` and where the direction `v` has curvature `0`.
"""
function tangent_orthonormal_basis(M::MetricManifold{SymmetricPositiveDefinite{n},LinearAffineMetric},x,v) where n
    xSqrt = sqrt(x) 
    V = eigvecs(v)
    Ξ = [ (i==j ? 1/2 : 1/sqrt(2))*( V[:,i] * transpose(V[:,j])  +  V[:,j] * transpose(V[:,i]) )
        for i=1:n for j= i:n
    ]
    λ = eigvals(v)
    κ = [ -1/4 * (λ[i]-λ[j])^2 for i=1:n for j= i:n ]
  return Ξ,κ
end

@doc doc"""
    injectivity_radius(P)

return the injectivity radius of the [`SymmetricPositiveDefinite`](@ref). Since `P`  is a Hadamard manifold,
the injectivity radius is $\infty$.
"""
injectivity_radius(P::SymmetricPositiveDefinite, args...) = Inf

@doc doc"""
    zero_tangent_vector(P,x)

returns the zero tangent vector in the tangent space of the symmetric positive
definite matrix `x` on the [`SymmetricPositiveDefinite`](@ref) manifold `P`.
"""
zero_tangent_vector(P::SymmetricPositiveDefinite, x) = zero(x)

@doc doc"""
    zero_tangent_vector(P,x)

returns the zero tangent vector in the variable `v` from the tangent space of
the symmetric positive definite matrix `x` on
the [`SymmetricPositiveDefinite`](@ref) manifold `P`.
"""
zero_tangent_vector!(P::SymmetricPositiveDefinite, v, x) = fill!(v, 0)

"""
    is_manifold_point(S,x; kwargs...)

checks, whether `x` is a valid point on the [`SymmetricPositiveDefinite{N}`](@ref) `P`, i.e. is a matrix
of size `(N,N)`, symmetric and positive definite.
The tolerance for the second to last test can be set using the ´kwargs...`.
"""
function is_manifold_point(P::SymmetricPositiveDefinite{N},x; kwargs...) where N
    if size(x) != representation_size(P)
        throw(DomainError(size(x),"The point $(x) does not lie on $(P), since its size is not $(representation_size(P))."))
    end
    if !isapprox(norm(x-transpose(x)), 0.; kwargs...)
        throw(DomainError(norm(x), "The point $(x) does not lie on $(P) since its not a symmetric matrix:"))
    end
    if ! all( eigvals(x) .> 0 )
        throw(DomainError(norm(x), "The point $x does not lie on $P since its not a positive definite matrix."))
    end
    return true
end

"""
    is_tangent_vector(S,x,v; kwargs... )

checks whether `v` is a tangent vector to `x` on the [`SymmetricPositiveDefinite`](@ref) `S`, i.e.
atfer [`is_manifold_point`](@ref)`(S,x)`, `v` has to be of same dimension as `x`
and a symmetric matrix, i.e. this stores tangent vetors as elements of the corresponding Lie group.
The tolerance for the last test can be set using the ´kwargs...`.
"""
function is_tangent_vector(P::SymmetricPositiveDefinite{N},x,v; kwargs...) where N
    is_manifold_point(P,x)
    if size(v) != representation_size(P)
        throw(DomainError(size(v),
            "The vector $(v) is not a tangent to a point on $(P) since its size does not match $(representation_size(P))."))
    end
    if !isapprox(norm(v-transpose(v)), 0.; kwargs...)
        throw(DomainError(size(v),
            "The vector $(v) is not a tangent to a point on $(P) (represented as an element of the Lie algebra) since its not symmetric."))
    end
    return true
end