@doc raw"""
    Euclidean{T<:Tuple,𝔽} <: AbstractManifold{𝔽}

Euclidean vector space.

# Constructor

    Euclidean(n)

Generate the ``n``-dimensional vector space ``ℝ^n``.

    Euclidean(n₁,n₂,...,nᵢ; field=ℝ)
    𝔽^(n₁,n₂,...,nᵢ) = Euclidean(n₁,n₂,...,nᵢ; field=𝔽)

Generate the vector space of ``k = n_1 \cdot n_2 \cdot … \cdot n_i`` values, i.e. the
manifold ``𝔽^{n_1, n_2, …, n_i}``, ``𝔽\in\{ℝ,ℂ\}``, whose
elements are interpreted as ``n_1 × n_2 × … × n_i`` arrays.
For ``i=2`` we obtain a matrix space.
The default `field=ℝ` can also be set to `field=ℂ`.
The dimension of this space is ``k \dim_ℝ 𝔽``, where ``\dim_ℝ 𝔽`` is the
[`real_dimension`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.real_dimension-Tuple{ManifoldsBase.AbstractNumbers}) of the field ``𝔽``.

    Euclidean(; field=ℝ)

Generate the 1D Euclidean manifold for an `ℝ`-, `ℂ`-valued  real- or complex-valued immutable
values (in contrast to 1-element arrays from the constructor above).
"""
struct Euclidean{N,𝔽} <: AbstractDecoratorManifold{𝔽} where {N<:Tuple} end

function Euclidean(n::Vararg{Int,I}; field::AbstractNumbers=ℝ) where {I}
    return Euclidean{Tuple{n...},field}()
end

function active_traits(f, ::Euclidean, args...)
    return merge_traits(
        IsDefaultMetric(EuclideanMetric()),
        IsDefaultConnection(LeviCivitaConnection()),
    )
end

Base.:^(𝔽::AbstractNumbers, n) = Euclidean(n...; field=𝔽)

"""
    EuclideanMetric <: RiemannianMetric

A general type for any manifold that employs the Euclidean Metric, for example
the [`Euclidean`](@ref) manifold itself, or the [`Sphere`](@ref), where every
tangent space (as a plane in the embedding) uses this metric (in the embedding).

Since the metric is independent of the field type, this metric is also used for
the Hermitian metrics, i.e. metrics that are analogous to the `EuclideanMetric`
but where the field type of the manifold is `ℂ`.

This metric is the default metric for example for the [`Euclidean`](@ref) manifold.
"""
struct EuclideanMetric <: RiemannianMetric end

Base.:^(M::Euclidean, n::Int) = ^(M, (n,))
function Base.:^(::Euclidean{T,𝔽}, n::NTuple{N,Int}) where {T,𝔽,N}
    return Euclidean{Tuple{T.parameters...,n...},𝔽}()
end

function allocation_promotion_function(
    ::Euclidean{<:Tuple,ℂ},
    ::Union{typeof(get_vector),typeof(get_coordinates)},
    ::Tuple,
)
    return complex
end

function check_point(M::Euclidean{N,𝔽}, p) where {N,𝔽}
    if (𝔽 === ℝ) && !(eltype(p) <: Real)
        return DomainError(
            eltype(p),
            "The matrix $(p) is not a real-valued matrix, so it does not lie on $(M).",
        )
    end
    if (𝔽 === ℂ) && !(eltype(p) <: Real) && !(eltype(p) <: Complex)
        return DomainError(
            eltype(p),
            "The matrix $(p) is neither a real- nor complex-valued matrix, so it does not lie on $(M).",
        )
    end
    return nothing
end

function check_vector(M::Euclidean{N,𝔽}, p, X; kwargs...) where {N,𝔽}
    if (𝔽 === ℝ) && !(eltype(X) <: Real)
        return DomainError(
            eltype(X),
            "The matrix $(X) is not a real-valued matrix, so it can not be a tangent vector to $(p) on $(M).",
        )
    end
    if (𝔽 === ℂ) && !(eltype(X) <: Real) && !(eltype(X) <: Complex)
        return DomainError(
            eltype(X),
            "The matrix $(X) is neither a real- nor complex-valued matrix, so it can not be a tangent vector to $(p) on $(M).",
        )
    end
    return nothing
end

function det_local_metric(
    ::MetricManifold{𝔽,<:AbstractManifold,EuclideanMetric},
    p,
    ::InducedBasis{𝔽,TangentSpaceType,<:RetractionAtlas},
) where {𝔽}
    return one(eltype(p))
end

"""
    distance(M::Euclidean, p, q)

Compute the Euclidean distance between two points on the [`Euclidean`](@ref)
manifold `M`, i.e. for vectors it's just the norm of the difference, for matrices
and higher order arrays, the matrix and ternsor Frobenius norm, respectively.
"""
distance(::Euclidean, p, q) = norm(p .- q)
distance(::Euclidean{Tuple{}}, p::Number, q::Number) = abs(p - q)

"""
    embed(M::Euclidean, p)

Embed the point `p` in `M`. Equivalent to an identity map.
"""
embed(::Euclidean, p) = p

"""
    embed(M::Euclidean, p, X)

Embed the tangent vector `X` at point `p` in `M`. Equivalent to an identity map.
"""
embed(::Euclidean, p, X) = X

function embed!(
    ::EmbeddedManifold{𝔽,Euclidean{nL,𝔽},Euclidean{mL,𝔽2}},
    q,
    p,
) where {nL,mL,𝔽,𝔽2}
    n = size(p)
    ln = length(n)
    m = size(q)
    lm = length(m)
    (length(n) > length(m)) && throw(
        DomainError(
            "Invalid embedding, since Euclidean dimension ($(n)) is longer than embedding dimension $(m).",
        ),
    )
    any(n .> m[1:ln]) && throw(
        DomainError(
            "Invalid embedding, since Euclidean dimension ($(n)) has entry larger than embedding dimensions ($(m)).",
        ),
    )
    # put p into q
    fill!(q, 0)
    # fill „top left edge“ of q with p.
    q[map(ind_n -> Base.OneTo(ind_n), n)..., ntuple(_ -> 1, lm - ln)...] .= p
    return q
end

@doc raw"""
    exp(M::Euclidean, p, X)

Compute the exponential map on the [`Euclidean`](@ref) manifold `M` from `p` in direction
`X`, which in this case is just
````math
\exp_p X = p + X.
````
"""
Base.exp(::Euclidean, ::Any...)
Base.exp(::Euclidean, p::Number, q::Number) = p + q

exp!(::Euclidean, q, p, X) = (q .= p .+ X)

function get_basis_diagonalizing(
    M::Euclidean,
    p,
    B::DiagonalizingOrthonormalBasis{𝔽},
) where {𝔽}
    vecs = get_vectors(M, p, get_basis(M, p, DefaultOrthonormalBasis(𝔽)))
    eigenvalues = zeros(real(eltype(p)), manifold_dimension(M))
    return CachedBasis(B, DiagonalizingBasisData(B.frame_direction, eigenvalues, vecs))
end

function get_coordinates_orthonormal!(M::Euclidean, c, p, X, ::RealNumbers)
    S = representation_size(M)
    PS = prod(S)
    copyto!(c, reshape(X, PS))
    return c
end

function get_coordinates_induced_basis!(
    M::Euclidean,
    c,
    p,
    X,
    ::InducedBasis{ℝ,TangentSpaceType,<:RetractionAtlas},
)
    S = representation_size(M)
    PS = prod(S)
    copyto!(c, reshape(X, PS))
    return c
end

function get_coordinates_orthonormal!(
    M::Euclidean{<:Tuple,ℂ},
    c,
    ::Any,
    X,
    ::ComplexNumbers,
)
    S = representation_size(M)
    PS = prod(S)
    c .= [reshape(real.(X), PS)..., reshape(imag(X), PS)...]
    return c
end

function get_coordinates_diagonalizing!(
    M::Euclidean{<:Tuple,ℂ},
    c,
    ::Any,
    X,
    ::DiagonalizingOrthonormalBasis{ℂ},
)
    S = representation_size(M)
    PS = prod(S)
    c .= [reshape(real.(X), PS)..., reshape(imag(X), PS)...]
    return c
end
function get_coordinates_diagonalizing!(
    M::Euclidean,
    c,
    p,
    X,
    ::DiagonalizingOrthonormalBasis{ℝ},
) where {𝔽}
    S = representation_size(M)
    PS = prod(S)
    copyto!(c, reshape(X, PS))
    return c
end

function get_vector_orthonormal!(M::Euclidean, Y, ::Any, c, ::RealNumbers)
    S = representation_size(M)
    copyto!(Y, reshape(c, S))
    return Y
end
function get_vector_diagonalizing!(
    M::Euclidean,
    Y,
    ::Any,
    c,
    B::DiagonalizingOrthonormalBasis,
)
    S = representation_size(M)
    copyto!(Y, reshape(c, S))
    return Y
end
function get_vector_induced_basis!(M::Euclidean, Y, ::Any, c, B::InducedBasis)
    S = representation_size(M)
    copyto!(Y, reshape(c, S))
    return Y
end
function get_vector_orthonormal!(M::Euclidean{<:Tuple,ℂ}, Y, ::Any, c, ::ComplexNumbers)
    S = representation_size(M)
    N = div(length(c), 2)
    copyto!(Y, reshape(c[1:N] + im * c[(N + 1):end], S))
    return Y
end
function get_vector_diagonalizing!(
    M::Euclidean{<:Tuple,ℂ},
    Y,
    ::Any,
    c,
    ::DiagonalizingOrthonormalBasis{ℂ},
)
    S = representation_size(M)
    N = div(length(c), 2)
    copyto!(Y, reshape(c[1:N] + im * c[(N + 1):end], S))
    return Y
end
@doc raw"""
    injectivity_radius(M::Euclidean)

Return the injectivity radius on the [`Euclidean`](@ref) `M`, which is ``∞``.
"""
injectivity_radius(::Euclidean) = Inf

@doc raw"""
    inner(M::Euclidean, p, X, Y)

Compute the inner product on the [`Euclidean`](@ref) `M`, which is just
the inner product on the real-valued or complex valued vector space
of arrays (or tensors) of size ``n_1 × n_2  ×  …  × n_i``, i.e.

````math
g_p(X,Y) = \sum_{k ∈ I} \overline{X}_{k} Y_{k},
````

where ``I`` is the set of vectors ``k ∈ ℕ^i``, such that for all

``i ≤ j ≤ i`` it holds ``1 ≤ k_j ≤ n_j`` and ``\overline{\cdot}`` denotes the complex conjugate.

For the special case of ``i ≤ 2``, i.e. matrices and vectors, this simplifies to

````math
g_p(X,Y) = X^{\mathrm{H}}Y,
````

where ``\cdot^{\mathrm{H}}`` denotes the Hermitian, i.e. complex conjugate transposed.
"""
inner(::Euclidean, ::Any...)
@inline inner(::Euclidean, p, X, Y) = dot(X, Y)
@inline function inner(
    ::MetricManifold{𝔽,<:AbstractManifold,EuclideanMetric},
    p,
    X,
    Y,
) where {𝔽}
    return dot(X, Y)
end

function inverse_local_metric(
    M::MetricManifold{𝔽,<:AbstractManifold,EuclideanMetric},
    p,
    B::InducedBasis{𝔽,TangentSpaceType,<:RetractionAtlas},
) where {𝔽}
    return local_metric(M, p, B)
end
function inverse_local_metric(
    M::Euclidean,
    p,
    B::InducedBasis{𝔽,TangentSpaceType,<:RetractionAtlas},
) where {𝔽}
    return local_metric(M, p, B)
end

function local_metric(
    ::MetricManifold{𝔽,<:AbstractManifold,EuclideanMetric},
    p,
    B::InducedBasis{𝔽,TangentSpaceType,<:RetractionAtlas},
) where {𝔽}
    return Diagonal(ones(SVector{size(p, 1),eltype(p)}))
end
function local_metric(
    ::Euclidean,
    p,
    B::InducedBasis{𝔽,TangentSpaceType,<:RetractionAtlas},
) where {𝔽}
    return Diagonal(ones(SVector{size(p, 1),eltype(p)}))
end

@doc raw"""
    log(M::Euclidean, p, q)

Compute the logarithmic map on the [`Euclidean`](@ref) `M` from `p` to `q`,
which in this case is just
````math
\log_p q = q-p.
````
"""
Base.log(::Euclidean, ::Any...)
Base.log(::Euclidean{Tuple{}}, p::Number, q::Number) = q - p
Base.log(::Euclidean, p, q) = q .- p

log!(::Euclidean, X, p, q) = (X .= q .- p)

function log_local_metric_density(
    ::MetricManifold{𝔽,<:AbstractManifold,EuclideanMetric},
    p,
    ::InducedBasis{𝔽,TangentSpaceType,<:RetractionAtlas},
) where {𝔽}
    return zero(eltype(p))
end

@generated _product_of_dimensions(::Euclidean{N}) where {N} = prod(N.parameters)

"""
    manifold_dimension(M::Euclidean)

Return the manifold dimension of the [`Euclidean`](@ref) `M`, i.e.
the product of all array dimensions and the [`real_dimension`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.real_dimension-Tuple{ManifoldsBase.AbstractNumbers}) of the
underlying number system.
"""
function manifold_dimension(M::Euclidean{N,𝔽}) where {N,𝔽}
    return _product_of_dimensions(M) * real_dimension(𝔽)
end
manifold_dimension(::Euclidean{Tuple{},𝔽}) where {𝔽} = real_dimension(𝔽)

Statistics.mean(::Euclidean{Tuple{}}, x::AbstractVector{<:Number}; kwargs...) = mean(x)
function Statistics.mean(
    ::Euclidean{Tuple{}},
    x::AbstractVector{<:Number},
    w::AbstractWeights;
    kwargs...,
)
    return mean(x, w)
end
Statistics.mean(::Euclidean, x::AbstractVector; kwargs...) = mean(x)

function StatsBase.mean_and_var(
    ::Euclidean{Tuple{}},
    x::AbstractVector{<:Number};
    kwargs...,
)
    m, v = mean_and_var(x; kwargs...)
    return m, sum(v)
end
function StatsBase.mean_and_var(
    ::Euclidean{Tuple{}},
    x::AbstractVector{<:Number},
    w::AbstractWeights;
    corrected=false,
    kwargs...,
)
    m, v = mean_and_var(x, w; corrected=corrected, kwargs...)
    return m, sum(v)
end

Statistics.median(::Euclidean{Tuple{}}, x::AbstractVector{<:Number}; kwargs...) = median(x)
function Statistics.median(
    ::Euclidean{Tuple{}},
    x::AbstractVector{<:Number},
    w::AbstractWeights;
    kwargs...,
)
    return median(x, w)
end

mid_point(::Euclidean, p1, p2) = (p1 .+ p2) ./ 2
mid_point(::Euclidean{Tuple{}}, p1::Number, p2::Number) = (p1 + p2) / 2

function mid_point!(::Euclidean, q, p1, p2)
    q .= (p1 .+ p2) ./ 2
    return q
end

@doc raw"""
    norm(M::Euclidean, p, X)

Compute the norm of a tangent vector `X` at `p` on the [`Euclidean`](@ref)
`M`, i.e. since every tangent space can be identified with `M` itself
in this case, just the (Frobenius) norm of `X`.
"""
LinearAlgebra.norm(::Euclidean, ::Any, X) = norm(X)
LinearAlgebra.norm(::MetricManifold{ℝ,<:AbstractManifold,EuclideanMetric}, p, X) = norm(X)

function project!(
    ::EmbeddedManifold{𝔽,Euclidean{nL,𝔽},Euclidean{mL,𝔽2}},
    q,
    p,
) where {nL,mL,𝔽,𝔽2}
    n = size(p)
    ln = length(n)
    m = size(q)
    lm = length(m)
    (length(n) < length(m)) && throw(
        DomainError(
            "Invalid embedding, since Euclidean dimension ($(n)) is longer than embedding dimension $(m).",
        ),
    )
    any(n .< m[1:ln]) && throw(
        DomainError(
            "Invalid embedding, since Euclidean dimension ($(n)) has entry larger than embedding dimensions ($(m)).",
        ),
    )
    #  fill q with the „top left edge“ of p.
    q .= p[map(i -> Base.OneTo(i), m)..., ntuple(_ -> 1, lm - ln)...]
    return q
end

"""
    parallel_transport_along(M::Euclidean, p, X, c)

the parallel transport on [`Euclidean`](@ref) is the identiy, i.e. returns `X`.
"""
parallel_transport_along(::Euclidean, ::Any, X, c::AbstractVector) = X
parallel_transport_along!(::Euclidean, Y, ::Any, X, c::AbstractVector) = copyto!(Y, X)

"""
    parallel_transport_direction(M::Euclidean, p, X, d)

the parallel transport on [`Euclidean`](@ref) is the identiy, i.e. returns `X`.
"""
parallel_transport_direction(::Euclidean, ::Any, X, ::Any) = X
parallel_transport_direction!(::Euclidean, Y, ::Any, X, ::Any) = copyto!(Y, X)

"""
    parallel_transport_to(M::Euclidean, p, X, q)

the parallel transport on [`Euclidean`](@ref) is the identiy, i.e. returns `X`.
"""
parallel_transport_to(::Euclidean, ::Any, X, ::Any) = X
parallel_transport_to!(::Euclidean, Y, ::Any, X, ::Any) = copyto!(Y, X)

@doc raw"""
    project(M::Euclidean, p)

Project an arbitrary point `p` onto the [`Euclidean`](@ref) manifold `M`, which
is of course just the identity map.
"""
project(::Euclidean, ::Any)
project(::Euclidean{Tuple{}}, p::Number) = p

project!(::Euclidean, q, p) = copyto!(q, p)

"""
    project(M::Euclidean, p, X)

Project an arbitrary vector `X` into the tangent space of a point `p` on the
[`Euclidean`](@ref) `M`, which is just the identity, since any tangent
space of `M` can be identified with all of `M`.
"""
project(::Euclidean, ::Any, ::Any)
project(::Euclidean{Tuple{}}, ::Number, X::Number) = X

project!(::Euclidean, Y, p, X) = copyto!(Y, X)

function Random.rand!(::Euclidean, pX; σ=one(eltype(pX)), vector_at=nothing)
    pX .= randn(size(pX)) .* σ
    return pX
end
function Random.rand!(
    rng::AbstractRNG,
    ::Euclidean,
    pX;
    σ=one(eltype(pX)),
    vector_at=nothing,
)
    pX .= randn(rng, size(pX)) .* σ
    return pX
end

"""
    representation_size(M::Euclidean)

Return the array dimensions required to represent an element on the
[`Euclidean`](@ref) `M`, i.e. the vector of all array dimensions.
"""
@generated representation_size(::Euclidean{N}) where {N} = size_to_tuple(N)
@generated representation_size(::Euclidean{Tuple{}}) = ()

function retract(M::Euclidean{Tuple{}}, p::Number, q::Number)
    return retract(M, p, q, ExponentialRetraction())
end
function retract(M::Euclidean{Tuple{}}, p::Number, q::Number, ::ExponentialRetraction)
    return exp(M, p, q)
end

function Base.show(io::IO, ::Euclidean{N,𝔽}) where {N,𝔽}
    return print(io, "Euclidean($(join(N.parameters, ", ")); field = $(𝔽))")
end
#
# Vector Transport
#
# The following functions are defined on layer 1 already, since
# a) its independent of the transport or retraction method
# b) no amibuities occur
# c) Euclidean is so basic, that these are plain defaults
#
function vector_transport_along(
    ::Euclidean,
    ::Any,
    X,
    ::AbstractVector,
    method::AbstractVectorTransportMethod,
)
    return X
end
function vector_transport_along!(
    M::Euclidean,
    Y,
    ::Any,
    X,
    ::AbstractVector,
    ::AbstractVectorTransportMethod=default_vector_transport_method(M),
)
    return copyto!(Y, X)
end
function vector_transport_direction(
    M::Euclidean,
    ::Any,
    X,
    ::Any,
    ::AbstractVectorTransportMethod=default_vector_transport_method(M),
    ::AbstractRetractionMethod=default_retraction_method(M),
)
    return X
end
function vector_transport_direction!(
    M::Euclidean,
    Y,
    ::Any,
    X,
    ::Any,
    ::AbstractVectorTransportMethod=default_vector_transport_method(M),
    ::AbstractRetractionMethod=default_retraction_method(M),
)
    return copyto!(Y, X)
end
"""
    vector_transport_to(M::Euclidean, p, X, q, ::AbstractVectorTransportMethod)

Transport the vector `X` from the tangent space at `p` to the tangent space at `q`
on the [`Euclidean`](@ref) `M`, which simplifies to the identity.
"""
vector_transport_to(::Euclidean, ::Any, ::Any, ::Any, ::AbstractVectorTransportMethod)
function vector_transport_to(
    M::Euclidean,
    ::Any,
    X,
    ::Any,
    ::AbstractVectorTransportMethod=default_vector_transport_method(M),
    ::AbstractRetractionMethod=default_retraction_method(M),
)
    return X
end

function vector_transport_to!(
    M::Euclidean,
    Y,
    ::Any,
    X,
    ::Any,
    ::AbstractVectorTransportMethod=default_vector_transport_method(M),
    ::AbstractRetractionMethod=default_retraction_method(M),
)
    return copyto!(Y, X)
end

Statistics.var(::Euclidean, x::AbstractVector; kwargs...) = sum(var(x; kwargs...))
function Statistics.var(::Euclidean, x::AbstractVector{<:Number}, m::Number; kwargs...)
    return sum(var(x; mean=m, kwargs...))
end

"""
    zero_vector(M::Euclidean, x)

Return the zero vector in the tangent space of `x` on the [`Euclidean`](@ref)
`M`, which here is just a zero filled array the same size as `x`.
"""
zero_vector(::Euclidean, ::Any...)
zero_vector(::Euclidean{Tuple{}}, p::Number) = zero(p)

zero_vector!(::Euclidean, v, ::Any) = fill!(v, 0)
