using LinearAlgebra: svd, eig
@doc doc"""
    SymmetricPositiveDefinite{N} <: Manifold

The manifold of symmetric positive definite matrices, i.e.

```math
\mathcal P(n) \coloneqq
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
@generated manifold_dimension(::SymmetricPositiveDefinite{N}) where {N} = N*(N+1)/2

@doc doc"""

"""
struct LinearAffineMetric <: Metric end
@traitimpl HasMetric{SymmetricPositiveDefinite, LogEuclidean}

@doc doc"""

"""
struct LogEuclideanMetric <: Metric end
@traitimpl HasMetric{SymmetricPositiveDefinite, LogEuclidean}

distance(P::SymmetricPositiveDefinite{N},x,y) = distance(MetricManifold(S,LinearAffineMetric)},x,y)
function distance(P::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric},x,y)
    s = real.( eigen( x,y ).values )
    return any(s .<= eps() ) ? 0 : sqrt(  sum( abs.(log.(s)).^2 )  )
end

inner(P::SymmetricPositiveDefinite{N}, x, w, v) = inner(MetricManifold(S,LinearAffineMetric),x,w,v)
function inner(P::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric},x,w,v)
	svd1 = svd(x)
	U = svd1.U
	S = svd1.S
	SInv = Diagonal( 1 ./ S )
	return tr( w * U * SInv * transpose(U) * v * U * SInv * transpose(U) )
end

norm(P::SymmetricPositiveDefinite{N},x,v) = sqrt( inner(P,x,v,v) )
norm(P::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric},x,v) = sqrt( inner(P,x,v,v) )

function exp!(P::SymmetricPositiveDefinite{N},y,x,v) = exp!(MetricManifold(SymmetricPositiveDefinite,LinearAffineMetric),y,x,v)
function exp!(P::Metricmanifold{SymmetricPositiveDefinite{N},LinearAffineMetric}, y, x, v)
	svd1 = svd(x)
	U = svd1.U
	S = svd1.S
    Ssqrt = Diagonal( sqrt.(S) )
    SsqrtInv = Diagonal( 1 ./ sqrt.(S) )
    xSqrt = U*Ssqrt*transpose(U);
    xSqrtInv = U*SsqrtInv*transpose(U)
    T = xSqrtInv * (t.*ξ.value) * xSqrtInv
    eig1 = eigen(0.5*( T + transpose(T) ) ) # numerical stabilization
   	Se = Diagonal( exp.(eig1.values) )
    Ue = eig1.vectors
    y = xSqrt*Ue*Se*transpose(Ue)*xSqrt
    y = 0.5*( y + transpose(y) ) ) # numerical stabilization
    return y
end

function log!(P::SymmetricPositiveDefinite{N}, v, x, y) = exp!(MetricManifold(SymmetricPositiveDefinite,LinearAffineMetric),y,x,v)
function log!(P::Metricmanifold{SymmetricPositiveDefinite{N},LinearAffineMetric}, v, x, y)
    svd1 = svd( x )
	U = svd1.U
	S = svd1.S
    Ssqrt = Diagonal( sqrt.(S) )
    SsqrtInv = Diagonal( 1 ./ sqrt.(S) )
    xSqrt = U*Ssqrt*transpose(U)
    xSqrtInv = U*SsqrtInv*transpose(U)
    T = xSqrtInv * getValue(y) * xSqrtInv
	svd2 = svd(0.5*(T+transpose(T)))
	Se = Diagonal( log.(max.(svd2.S,eps()) ) )
	Ue = svd2.U
	v = xSqrt * Ue*Se*transpose(Ue) * xSqrt
	v = 0.5*( v + transpose(v) ) )
    return v
end

injectivity_radius(P::SymmetricPositiveDefinite, args...) = Infπ

zero_tangent_vector(P::SymmetricPositiveDefinite, x) = zero(x)
zero_tangent_vector(P::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric},x) = zero(x)
zero_tangent_vector(P::MetricManifold{SymmetricPositiveDefinite{N},LogEuclidean},x) = zero(x)

zero_tangent_vector!(P::SymmetricPositiveDefinite, v, x) = (v .= zero(x))
zero_tangent_vector!(P::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric},v, x) = (v .= zero(x))
zero_tangent_vector!(P::MetricManifold{SymmetricPositiveDefinite{N},LogEuclidean},v, x) = (v .= zero(x))

"""
    is_manifold_point(S,x; kwargs...)

checks, whether `x` is a valid point on the [`SymmetricPositiveDefinite{N}`](@ref) `P`, i.e. is a matrix
of size `(N,N)`, symmetric and positive definite.
The tolerance for the second to last test can be set using the ´kwargs...`.
"""
function is_manifold_point(P::SymmetricPositiveDefinite{N},x; kwargs...) where {N}
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

checks whether `v` is a tangent vector to `x` on the [`Sphere`](@ref) `S`, i.e.
atfer [`is_manifold_point`](@ref)`(S,x)`, `v` has to be of same dimension as `x`
and orthogonal to `x`.
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