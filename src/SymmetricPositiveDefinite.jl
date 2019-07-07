using LinearAlgebra: svd, eigen

export SymmetricPositiveDefinite, LinearAffineMetric, LogEuclideanMetric

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
    distance(lP,x,y)

computes the distance on the [`SymmetricPositiveDefinite`](@ref) manifold between `x` and `y`,
as a [`MetricManifold`](@ref) with [`LinearAffineMetric`](@ref). The formula reads

```math
d_{\mathcal P(n)}(x,y) = \lVert \operatorname{Log}(x^{-\frac{1}{2}}yx^{-\frac{1}{2}})\rVert.
```
"""
function distance(P::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric},x,y) where N
    s = real.( eigen( x,y ).values )
    return any(s .<= eps() ) ? 0 : sqrt(  sum( abs.(log.(s)).^2 )  )
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
    return tr( (w / F) * (v / F ))
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
    xSqrt = U*Ssqrt*transpose(U);
    xSqrtInv = U*SsqrtInv*transpose(U)
    T = xSqrtInv * v * xSqrtInv
    eig1 = eigen( T ) # numerical stabilization
    Se = Diagonal( exp.(eig1.values) )
    Ue = eig1.vectors
    y = xSqrt*Ue*Se*transpose(Ue)*xSqrt
    return y
end

@doc doc"""
    log!(P,v,x,y)

compute the logarithmic map at `x` to `y` on the [`SymmetricPositiveDefinite`](@ref)
manifold with its default metric, [`LinearAffineMetric`](@ref) and modify `v`.
"""
log!(P::SymmetricPositiveDefinite{N}, v, x, y) where N = log!(MetricManifold(P,LinearAffineMetric()),y,x,v)
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
    Ssqrt = Diagonal( sqrt.(S) )
    SsqrtInv = Diagonal( 1 ./ sqrt.(S) )
    xSqrt = U*Ssqrt*transpose(U)
    xSqrtInv = U*SsqrtInv*transpose(U)
    T = xSqrtInv * y * xSqrtInv
    e2 = eigen( (T + transpose(T))/2 )
    Se = Diagonal( log.(max.(e2.values,eps()) ) )
    Ue = e2.vectors
    v = xSqrt * Ue*Se*transpose(Ue) * xSqrt
    return v
end

function representation_size(::SymmetricPositiveDefinite{N}, ::Type{T}) where {N, T<:Union{MPoint, TVector, CoTVector}}
    return (N,N)
end


@doc doc"""
    vector_transport(P,vto,x,v,y,::ParallelTransport)

compute the parallel transport on the [`SymmetricPositiveDefinite`](@ref) with its default metric, [`LinearAffineMetric`](@ref).
"""
vector_transport!(P::SymmetricPositiveDefinite{N},vto, x, v, y, m) where N = vector_transport!(MetricManifold(P,LinearAffineMetric()),vto, x, v, y, m)
@doc doc"""
    vector_transport!(P,vto,x,v,y,::ParallelTransport)

compute the parallel transport on the [`SymmetricPositiveDefinite`](@ref) as a [`MetricManifold`](@ref) with the [`LinearAffineMetric`](@ref).
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
and `log` the logarithmic map.
"""
function vector_transport!(::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric}, vto, x, v, y, ::ParallelTransport) where N
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
    xSqrt = U*Ssqrt*transpose(U)
    xSqrtInv = U*SsqrtInv*transpose(U)
    tv = xSqrtInv * v * xSqrtInv
    ty = xSqrtInv * y * xSqrtInv
    e2 = eigen( ty )
    Se = Diagonal( log.(e2.values) )
    Ue = e2.vectors
    ty2 = Ue*Se*transpose(Ue)
    e3 = eigen( ty2 )
    Sf = Diagonal( exp.(e3.values) )
    Uf = e3.vectors
    vto = xSqrt*Uf*Sf*transpose(Uf)*(0.5*(tv+transpose(tv)))*Uf*Sf*transpose(Uf)*xSqrt
    return vto
end

@doc doc"""
    injectivity_radius(P)

return the injectivity radius of the [`SymmetricPositiveDefinite`](@ref). Since `P`  is a Hadamard manifold,
the injectivity radius is $\infty$.
"""
injectivity_radius(P::SymmetricPositiveDefinite, args...) = Inf

zero_tangent_vector(P::SymmetricPositiveDefinite, x) = zero(x)
zero_tangent_vector!(P::SymmetricPositiveDefinite, v, x) = fill!(v, 0)

"""
    is_manifold_point(S,x; kwargs...)

checks, whether `x` is a valid point on the [`SymmetricPositiveDefinite{N}`](@ref) `P`, i.e. is a matrix
of size `(N,N)`, symmetric and positive definite.
The tolerance for the second to last test can be set using the ´kwargs...`.
"""
function is_manifold_point(P::SymmetricPositiveDefinite{N},x; kwargs...) where N
    if size(x) != (N,N)
        throw(DomainError(size(x),"The point $(x) does not lie on $P, since its size is not $( (N,N) )."))
    end
    if !isapprox(norm(x-transpose(x)), 0.; kwargs...)
        throw(DomainError(norm(x), "The point $x does not lie on the sphere $P since its not a symmetric matrix:"))
    end
    if ! all( eigvals(x) > 0 )
        throw(DomainError(norm(x), "The point $x does not lie on the sphere $P since its not a positive definite matrix."))
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
    if size(v) != (N,N)
        throw(DomainError(size(v),
            "The vector $(v) is not a tangent to a point on $S since its size does not match $( (N,N) )."))
    end
    if !isapprox(norm(v-transpose(v)), 0.; kwargs...)
        throw(DomainError(size(v),
            "The vector $(v) is not a tangent to a point on $S (represented as an element of the Lie algebrasince its not symmetric."))
    end
    return true
end