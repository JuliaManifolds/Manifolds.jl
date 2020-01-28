@doc doc"""
    Hyperbolic{N} <: Manifold

The hyperbolic space $ℍ^n$ represented by $n+1$-Tuples, i.e. in by
vectors in $ℝ^{n+1}$ using the Minkowsi metric, i.e.

```math
ℍ^n = \Bigl\{x\in ℝ^{n+1}
\ \Big|\ ⟨x,x⟩_{\mathrm{M}}= -x_{n+1}^2
+ \displaystyle\sum_{k=1}^n x_k^2 = -1, x_{n+1} > 0\Bigr\},
```

where $⟨·,·⟩_{\mathrm{M}}$ denotes the [`minkowski_dot`](@ref)
is Minkowski inner product. The tangent space $T_x ℍ^n$ is given by

````math
T_x ℍ^n \coloneqq \bigl\{
v \in ℝ^{n+1} \ \bigl |\ ⟨x,v⟩_{\mathrm{M}} = 0
\bigr\}.
````
The Minkowski inner product inntroduces the [`MinkowskiMetric`](@ref), which is
a Riemannian metric on the tangent bundle $T ℍ^n$.

# Constructor

    Hyperbolic(n)

Generate the $ℍ^{n}\subset ℝ^{n+1}$
"""
struct Hyperbolic{N} <: Manifold end

Hyperbolic(n::Int) = Hyperbolic{n}()

@doc doc"""
    MinkowskiMetric <: LorentzMetric

The Minkowski metric is a [`LorentzMetric`](@ref) with, i.e.

````math
⟨a,b⟩_{\mathrm{M}} = -a_{n+1}b_{n+1} +
\displaystyle\sum_{k=1}^n a_kb_k.
````
It is also the default metric e.g. for the [`Hyperbolic`](@ref) space.

!!! note
    While the `MinkowskiMetric` itself is not positive definite in the whole embedded space,
    it is positive definite when restricted to a tangent space $T_xℳ$,
    $x\in ℳ$, of the [`Hyperbolic`](@ref) space $ℳ$.
"""
struct MinkowskiMetric <: LorentzMetric end

"""
    check_manifold_point(M::Hyperbolic, x; kwargs...)

Check whether `x` is a valid point on the [`Hyperbolic`](@ref) `M`, i.e. is a vector with
[`minkowski_dot`](@ref) -1. The tolerance for the last test can be set using the `kwargs...`.
"""
function check_manifold_point(M::Hyperbolic, x; kwargs...)
    if size(x) != representation_size(M)
        return DomainError(
            size(x),
            "The point $(x) does not lie on $(M), since its size is not $(representation_size(M)).",
        )
    end
    if !isapprox(minkowski_dot(x, x), -1.0; kwargs...)
        return DomainError(
            minkowski_dot(x, x),
            "The point $(x) does not lie on $(M) since its Minkowski inner product is not -1.",
        )
    end
    return nothing
end

"""
    check_tangent_vector(M::Hyperbolic, x, v; kwargs... )

Check whether `v` is a tangent vector to `x` on the [`Hyperbolic`](@ref) `M`, i.e.
after [`check_manifold_point`](@ref)`(M,x)`, `v` has to be of same dimension as `x`
and orthogonal to `x` with respect to [`minkowski_dot`](@ref).
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_tangent_vector(M::Hyperbolic, x, v; kwargs...)
    perr = check_manifold_point(M, x)
    perr === nothing || return perr
    if size(v) != representation_size(M)
        return DomainError(
            size(v),
            "The vector $(v) is not a tangent to a point on $M since its size does not match $(representation_size(M)).",
        )
    end
    if !isapprox(minkowski_dot(x, v), 0.0; kwargs...)
        return DomainError(
            abs(minkowski_dot(x, v)),
            "The vector $(v) is not a tangent vector to $(x) on $(M), since it is not orthogonal (with respect to the Minkowski inner product) in the embedding.",
        )
    end
    return nothing
end

@doc doc"""
    distance(M::Hyperbolic, x, y)

Compute the distance on the [`Hyperbolic`](@ref) `M`, which reads

````math
d_{ℍ^n}(x,y) = \operatorname{acosh}( - ⟨x, y⟩_{\mathrm{M}}),
````

where $⟨·,·⟩_{\mathrm{M}}$ denotes the [`minkowski_dot`](@ref).
"""
distance(M::Hyperbolic, x, y) = acosh(max(-minkowski_dot(x, y), 1.0))

@doc doc"""
    exp(M::Hyperbolic, x, v)

Compute the exponential map on the [`Hyperbolic`](@ref) space $ℍ^n$ eminating
from `x` towards `v`, which is optionally scaled by `t`. The formula reads

````math
\exp_x v = \cosh(\sqrt{⟨v,v⟩_{\mathrm{M}}})x
+ \sinh(\sqrt{⟨v,v⟩_{\mathrm{M}}})\frac{v}{\sqrt{⟨v,v⟩_{\mathrm{M}}}},
````

where $⟨·,·⟩_{\mathrm{M}}$ denotes the [`minkowski_dot`](@ref).
"""
exp(::Hyperbolic, ::Any...)

function exp!(M::Hyperbolic, y, x, v)
    vn = sqrt(max(minkowski_dot(v, v), 0.0))
    vn < eps(eltype(x)) && return copyto!(y, x)
    return copyto!(y, cosh(vn) * x + sinh(vn) / vn * v)
end

flat!(M::Hyperbolic, v::CoTFVector, x, w::TFVector) = copyto!(v, w)

@doc doc"""
    injectivity_radius(M::Hyperbolic[, x])

Return the injectivity radius on the [`Hyperbolic`](@ref), which is always $\infty$.
"""
injectivity_radius(H::Hyperbolic, args...) = Inf

@doc doc"""
    inner(M::Hyperbolic, x, v, w)

Compute the Riemannian inner product for two tangent vectors `v` and `w`
from $T_x ℍ^n$ of the [`Hyperbolic`](@ref) space $ℍ^n$ given by
$⟨w, v⟩_{\mathrm{M}}$ the [`minkowski_dot`](@ref) Minkowski
inner product on $ℝ^{n+1}$.
"""
@inline inner(M::Hyperbolic, x, w, v) = minkowski_dot(w, v)

is_default_metric(::Hyperbolic, ::MinkowskiMetric) = Val(true)

@doc doc"""
    log(M::Hyperbolic, x, y)

Compute the logarithmic map on the [`Hyperbolic`](@ref) space $ℍ^n$, the tangent
vector representing the [`geodesic`](@ref) starting from `x`
reaches `y` after time 1 on the [`Hyperbolic`](@ref) space `M`.
The formula reads for $x\neq y$

```math
\log_x y = d_{ℍ^n}(x,y)
\frac{y-⟨x,y⟩_{\mathrm{M}} x}{\lVert y-⟨x,y⟩_{\mathrm{M}} x \rVert_2}
```
and is zero otherwise.
"""
log(::Hyperbolic, ::Any...)

function log!(M::Hyperbolic, v, x, y)
    scp = minkowski_dot(x, y)
    w = y + scp * x
    wn = sqrt(max(scp .^ 2 - 1, 0.0))
    wn < eps(eltype(x)) && return zero_tangent_vector!(M, v, x)
    v .= acosh(max(1.0, -scp)) / wn .* w
    return v
end

@doc doc"""
    minkowski_dot(a,b)
Compute the Minkowski inner product of two Vectors `a` and `b` of same length
`n+1`, i.e.

````math
⟨a,b⟩_{\mathrm{M}} = -a_{n+1}b_{n+1} + \displaystyle\sum_{k=1}^n a_kb_k.
````
"""
function minkowski_dot(a::AbstractVector, b::AbstractVector)
    return -a[end] * b[end] + sum(a[1:end-1] .* b[1:end-1])
end

@doc doc"""
    manifold_dimension(H::Hyperbolic)

Return the dimension of the hyperbolic space manifold $ℍ^n$, i.e. $n$.
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
mean(::Hyperbolic, ::Any...)

function mean!(M::Hyperbolic, y, x::AbstractVector, w::AbstractVector; kwargs...)
    return mean!(M, y, x, w, CyclicProximalPointEstimation(); kwargs...)
end

@doc doc"""
    project_tangent(M::Hyperbolic, x, v)

Perform an orthogonal projection with respect to the Minkowski inner product of `v` onto
the tangent space at `x` of the [`Hyperbolic`](@ref) space `M`.

The formula reads
````math
w = v + ⟨x,v⟩_{\mathrm{M}} x,
````
where $⟨·, ·⟩_{\mathrm{M}}$ denotes the Minkowski inner
product in the embedding, see [`minkowski_dot`](@ref).
"""
project_tangent(::Hyperbolic, ::Any...)

project_tangent!(::Hyperbolic, w, x, v) = (w .= v .+ minkowski_dot(x, v) .* x)

@doc doc"""
    representation_size(M::Hyperbolic)

Return the representation size on the [`Hyperbolic`](@ref), i.e. for the `n`-diomensional
hyperbolic manifold the dimention of the embedding, i.e. `n+1`.
"""
@generated representation_size(::Hyperbolic{N}) where {N} = (N + 1,)

sharp!(M::Hyperbolic, v::TFVector, x, w::CoTFVector) = copyto!(v, w)

@doc doc"""
    vector_transport_to(M::Hyperbolic, x, v, y, ::ParallelTransport)

Compute the paralllel transport of the `v` from the tangent space at `x` on the
[`Hyperbolic`](@ref) space $ℍ^n$ to the tangent at `y` along the [`geodesic`](@ref)
connecting `x` and `y`. The formula reads

````math
𝒫_{y←x}(v) = v - \frac{⟨\log_xy,v⟩_x}{d^2_{ℍ^n}(x,y)}
\bigl(\log_xy + \log_yx \bigr).
````
"""
vector_transport_to(::Hyperbolic, ::Any, ::Any, ::Any, ::ParallelTransport)

function vector_transport_to!(M::Hyperbolic, vto, x, v, y, ::ParallelTransport)
    w = log(M, x, y)
    wn = norm(M, x, w)
    wn < eps(eltype(x + y)) && return copyto!(vto, v)
    return copyto!(vto, v - (inner(M, x, w, v) * (w + log(M, y, x)) / wn^2))
end

@doc doc"""
    zero_tangent_vector(M::Hyperbolic, x)

Return the zero vector from the tangent space at `x` of the [`Hyperbolic`](@ref) `M`.
"""
zero_tangent_vector(::HybridArray, ::Any...)

zero_tangent_vector!(M::Hyperbolic, v, x) = fill!(v, 0)
