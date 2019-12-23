@doc doc"""
    Hyperbolic{N} <: Manifold

The hyperbolic space $\mathbb H^n$ represented by $n+1$-Tuples, i.e. in by
vectors in $\mathbb R^{n+1}$ using the Minkowsi metric, i.e.

```math
\mathbb H^n = \Bigl\{x\in\mathbb R^{n+1}
\ \Big|\ \langle x,x \rangle_{\mathrm{M}}= -x_{n+1}^2
+ \displaystyle\sum_{k=1}^n x_k^2 = -1, x_{n+1} > 0\Bigr\},
```

where $\langle\cdot,\cdot\rangle_{\mathrm{M}}$ denotes the [`minkowski_dot`](@ref)
is Minkowski inner product, and this inner product in the embedded space yields
the Riemannian metric when restricted to the tangent bundle $T\mathbb H^n$.

# Constructor

    Hyperbolic(n)

generates the $\mathbb H^{n}\subset \mathbb R^{n+1}$
"""
struct Hyperbolic{N} <: Manifold end
Hyperbolic(n::Int) = Hyperbolic{n}()

"""
    check_manifold_point(S,x; kwargs...)

checks, whether `x` is a valid point on the [`Hyperbolic`](@ref) `M`, i.e. is a vector with
[`minkowski_dot`](@ref) -1. The tolerance for the last test can be set using the `kwargs...`.
"""
function check_manifold_point(M::Hyperbolic, x; kwargs...)
    if size(x) != representation_size(M)
        return DomainError(size(x),"The point $(x) does not lie on $(M), since its size is not $(representation_size(M)).")
    end
    if !isapprox(minkowski_dot(x,x), -1.; kwargs...)
        return DomainError(minkowski_dot(x,x), "The point $(x) does not lie on $(M) since its Minkowski inner product is not -1.")
    end
    return nothing
end

"""
    check_tangent_vector(M,x,v; kwargs... )

checks whether `v` is a tangent vector to `x` on the [`Hyperbolic`](@ref) `M`, i.e.
after [`check_manifold_point`](@ref)`(M,x)`, `v` has to be of same dimension as `x`
and orthogonal to `x` with respect to [`minkowski_dot`](@ref).
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_tangent_vector(M::Hyperbolic, x, v; kwargs...)
    perr = check_manifold_point(M,x)
    perr === nothing || return perr
    if size(v) != representation_size(M)
        return DomainError(size(v),
            "The vector $(v) is not a tangent to a point on $M since its size does not match $(representation_size(M)).")
    end
    if !isapprox( minkowski_dot(x,v), 0.; kwargs...)
        return DomainError(abs( minkowski_dot(x,v)),
            "The vector $(v) is not a tangent vector to $(x) on $(M), since it is not orthogonal (with respect to the Minkowski inner product) in the embedding."
        )
    end
    return nothing
end

distance(M::Hyperbolic, x, y) = acosh(max(-minkowski_dot(x, y), 1.))
@doc doc"""
    exp!(M, y, x, v[, t=1.0])

computes the exponential map on the [`Hyperbolic`](@ref) space $\mathbb H^n$ eminating from `x`
towards `v`, which is optionally scaled by `t`. The result is stored in `y`. The formula reads

````math
\exp_x v = \cosh(\sqrt{\langle v,v\rangle_{\mathrm{M}}})x
+ \sinh(\sqrt{\langle v,v\rangle_{\mathrm{M}}})\frac{v}{\sqrt{\langle v,v\rangle_{\mathrm{M}}}},
````
where $\langle\cdot,\cdot\rangle_{\mathrm{M}}$ denotes the [`minkowski_dot`](@ref).
"""
function exp!(M::Hyperbolic, y, x, v)
    vn = sqrt(max(minkowski_dot(v, v),0.))
    if vn < eps(eltype(x))
        y .= x
        return y
    end
    y .= cosh(vn)*x + sinh(vn)/vn*v
    return y
end

function flat!(M::Hyperbolic, v::FVector{CotangentSpaceType}, x, w::FVector{TangentSpaceType})
    copyto!(v.data, w.data)
    return v
end

injectivity_radius(H::Hyperbolic, args...) = Inf

@doc doc"""
    dot(M, x, v, w)

compute the Riemannian inner product for two tangent vectors `v` and `w`
from $T_x\mathcal M$ of the [`Hyperbolic`](@ref) space $\mathbb H^n$ given by
$\langle w, v \rangle_{\mathrm{M}}$ the [`minkowski_dot`](@ref) Minkowski
inner product on $\mathbb R^{n+1}$.
"""
@inline inner(M::Hyperbolic, x, w, v) = minkowski_dot(w, v)

@doc doc"""
    log!(M, v, x, y)

computes the logarithmic map on the [`Hyperbolic`](@ref) space $\mathbb H^n$, i.e., `v` 
corresponds to the tangent vector representing the [`geodesic`](@ref) starting from `x`
reaches `y` after time 1 on the [`Hyperbolic`](@ref) space `M`.
The formula reads for $x\neq y$

```math
\log_x y = d_{\mathbb H^n}(x,y)
\frac{y-\langle x,y\rangle_{\mathrm{M}} x}{\lVert y-\langle x,y\rangle_{\mathrm{M}} x \rVert_2}
```
and is zero otherwise.
"""
function log!(M::Hyperbolic, v, x, y)
    scp = minkowski_dot(x, y)
    w = y + scp*x
    wn = sqrt(max( scp.^2-1, 0.))
    if wn < eps(eltype(x))
        zero_tangent_vector!(M,v,x)
        return v
    end
    v .= acosh(max(1.,-scp))/wn .* w
    return v
end

@doc doc"""
    minkowski_dot(a,b)
computes the Minkowski inner product of two Vectors `a` and `b` of same length
`n+1`, i.e.

````math
\langle a,b\rangle_{\mathrm{M}} = -a_{n+1}b_{n+1} +
\displaystyle\sum_{k=1}^n a_kb_k.
````
"""
minkowski_dot(a::AbstractVector,b::AbstractVector) = -a[end]*b[end] + sum( a[1:end-1].*b[1:end-1] )

@doc doc"""
    manifold_dimension(H::Hyperbolic)

Return the dimension of the hyperbolic space manifold $\mathbb H^n$, i.e. $n$.
"""
manifold_dimension(::Hyperbolic{N}) where {N} = N

"""
    mean(
        M::Hyperbolic,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = CyclicProximalPointEstimationMethod();
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(M::Manifold, args...)) of `x` on the
[`Hyperbolic`](@ref) space using [`CyclicProximalPointEstimation`](@ref).
"""
mean(::Hyperbolic, args...)

mean!(M::Hyperbolic, y, x::AbstractVector, w::AbstractVector; kwargs...) =
    mean!(M, y, x, w,  CyclicProximalPointEstimation(); kwargs...)

@doc doc"""
    project_tangent!(M, w, x, v)

perform an orthogonal projection with respect to the Minkowski inner product of `v` onto
the tangent space at `x` of the [`Hyperbolic`](@ref) space `M`. The result is saved to `w`.

The formula reads
````math
w = v + \langle x,v\rangle_{\mathrm{M}} x,
````
where $\langle \cdot, \cdot \rangle_{\mathrm{M}}$ denotes the Minkowski inner
product in the embedding, see [`minkowski_dot`](@ref).
"""
project_tangent!(::Hyperbolic, w, x, v) = (w .= v .+ minkowski_dot(x, v) .* x)

representation_size(::Hyperbolic{N}) where {N} = (N+1,)

function sharp!(M::Hyperbolic, v::FVector{TangentSpaceType}, x, w::FVector{CotangentSpaceType})
    copyto!(v.data, w.data)
    return v
end

@doc doc"""
    vector_transport_to!(M, vto, x, v, y, ::ParallelTransport)

Compute the paralllel transport of the `v` from the tangent space at `x` on the
[`Hyperbolic`](@ref) space $\mathbb H^n$ to the tangent at `y` along the [`geodesic`](@ref)
connecting `x` and `y`. The formula reads

````math
P_{x\to y}(v) = v - \frac{\langle \log_xy,v\rangle_x}{d^2_{\mathbb H^n}(x,y)}
\bigl(\log_xy + \log_yx \bigr).
````
"""
function vector_transport_to!(M::Hyperbolic, vto, x, v, y, ::ParallelTransport)
    w = log(M,x,y)
    wn = norm(M,x,w)
    if (wn < eps(eltype(x+y)))
        copyto!(vto,v)
        return vto
    end
    vto .= v - (inner(M,x,w,v)*(w + log(M,y,x))/wn^2 )
    return vto
end

function zero_tangent_vector!(M::Hyperbolic, v, x)
    fill!(v, 0)
    return v
end