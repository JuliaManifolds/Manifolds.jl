@doc raw"""
    Euclidean{T<:Tuple,𝔽} <: Manifold{𝔽}

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
[`real_dimension`](@ref) of the field ``𝔽``.

    Euclidean(; field=ℝ)

Generate the 1D Euclidean manifold for an `ℝ`-, `ℂ`-valued  real- or complex-valued immutable
values (in contrast to 1-element arrays from the constructor above).
"""
struct Euclidean{N,𝔽} <: Manifold{𝔽} where {N<:Tuple} end

function Euclidean(n::Vararg{Int,I}; field::AbstractNumbers=ℝ) where {I}
    return Euclidean{Tuple{n...},field}()
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

function check_manifold_point(M::Euclidean{N,𝔽}, p) where {N,𝔽}
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
    if size(p) != representation_size(M)
        return DomainError(
            size(p),
            "The matrix $(p) does not lie on $(M), since its dimensions ($(size(p))) are wrong (expected: $(representation_size(M))).",
        )
    end
    return nothing
end

function check_tangent_vector(
    M::Euclidean{N,𝔽},
    p,
    X;
    check_base_point=true,
    kwargs...,
) where {N,𝔽}
    if check_base_point
        mpe = check_manifold_point(M, p; kwargs...)
        mpe === nothing || return mpe
    end
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
    if size(X) != representation_size(M)
        return DomainError(
            size(X),
            "The matrix $(X) does not lie in the tangent space of $(p) on $(M), since its dimensions $(size(X)) are wrong  (expected: $(representation_size(M))).",
        )
    end
    return nothing
end

function det_local_metric(::MetricManifold{𝔽,<:Manifold,EuclideanMetric}, p) where {𝔽}
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
embed(::Euclidean, p)
embed(::Euclidean{Tuple{}}, p) = p

"""
    embed(M::Euclidean, p, X)

Embed the tangent vector `X` at point `p` in `M`. Equivalent to an identity map.
"""
embed(::Euclidean, p, X)

embed!(::Euclidean, q, p) = copyto!(q, p)
embed!(::Euclidean, Y, p, X) = copyto!(Y, X)

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

"""
    flat(M::Euclidean, p, X)

Transform a tangent vector `X` into a cotangent. Since they can directly be identified in the
[`Euclidean`](@ref) case, this yields just the identity for a tangent vector `w` in the
tangent space of `p` on `M`.
"""
flat(::Euclidean, ::Any...)
function flat(::Euclidean{Tuple{}}, ::T, X::TFVector) where {T<:Number}
    return FVector(CotangentSpace, X.data)
end

flat!(::Euclidean, ξ::CoTFVector, p, X::TFVector) = copyto!(ξ, X)

function get_basis(::Euclidean, p, B::DefaultOrthonormalBasis{ℝ})
    vecs = [_euclidean_basis_vector(p, i) for i in eachindex(p)]
    return CachedBasis(B, vecs)
end
function get_basis(::Euclidean{<:Tuple,ℂ}, p, B::DefaultOrthonormalBasis{ℂ})
    vecs = [_euclidean_basis_vector(p, i) for i in eachindex(p)]
    return CachedBasis(B, [vecs; im * vecs])
end
function get_basis(M::Euclidean, p, B::DiagonalizingOrthonormalBasis)
    vecs = get_vectors(M, p, get_basis(M, p, DefaultOrthonormalBasis()))
    eigenvalues = zeros(real(eltype(p)), manifold_dimension(M))
    return CachedBasis(B, DiagonalizingBasisData(B.frame_direction, eigenvalues, vecs))
end

function get_coordinates!(M::Euclidean, Y, p, X, B::DefaultOrDiagonalizingBasis{ℝ})
    S = representation_size(M)
    PS = prod(S)
    copyto!(Y, reshape(X, PS))
    return Y
end
function get_coordinates!(
    M::Euclidean{<:Tuple,ℂ},
    Y,
    ::Any,
    X,
    ::DefaultOrDiagonalizingBasis{ℂ},
)
    S = representation_size(M)
    PS = prod(S)
    Y .= [reshape(real.(X), PS)..., reshape(imag(X), PS)...]
    return Y
end

function get_vector!(M::Euclidean, Y, ::Any, X, ::DefaultOrDiagonalizingBasis{ℝ})
    S = representation_size(M)
    Y .= reshape(X, S)
    return Y
end
function get_vector!(M::Euclidean{<:Tuple,ℂ}, Y, ::Any, X, ::DefaultOrDiagonalizingBasis{ℂ})
    S = representation_size(M)
    N = div(length(X), 2)
    Y .= reshape(X[1:N] + im * X[(N + 1):end], S)
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
@inline inner(::MetricManifold{𝔽,<:Manifold,EuclideanMetric}, p, X, Y) where {𝔽} = dot(X, Y)

function inverse_local_metric(M::MetricManifold{𝔽,<:Manifold,EuclideanMetric}, p) where {𝔽}
    return local_metric(M, p)
end

default_metric_dispatch(::Euclidean, ::EuclideanMetric) = Val(true)

function local_metric(::MetricManifold{𝔽,<:Manifold,EuclideanMetric}, p) where {𝔽}
    return Diagonal(ones(SVector{size(p, 1),eltype(p)}))
end

function inverse_retract(M::Euclidean{Tuple{}}, x::T, y::T) where {T<:Number}
    return inverse_retract(M, x, y, LogarithmicInverseRetraction())
end
function inverse_retract(
    M::Euclidean{Tuple{}},
    x::T,
    y::T,
    ::LogarithmicInverseRetraction,
) where {T<:Number}
    return log(M, x, y)
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
Base.log(::Euclidean{Tuple{}}, p::T, q::T) where {T<:Number} = q - p

log!(::Euclidean, X, p, q) = (X .= q .- p)

function log_local_metric_density(
    ::MetricManifold{𝔽,<:Manifold,EuclideanMetric},
    p,
) where {𝔽}
    return zero(eltype(p))
end

@generated _product_of_dimensions(::Euclidean{N}) where {N} = prod(N.parameters)

"""
    manifold_dimension(M::Euclidean)

Return the manifold dimension of the [`Euclidean`](@ref) `M`, i.e.
the product of all array dimensions and the [`real_dimension`](@ref) of the
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

function Statistics.mean!(M::Euclidean, p, x::AbstractVector, w::AbstractVector; kwargs...)
    return mean!(M, p, x, w, GeodesicInterpolation(); kwargs...)
end

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
function StatsBase.mean_and_var(
    M::Euclidean,
    x::AbstractVector,
    w::AbstractWeights;
    kwargs...,
)
    return mean_and_var(M, x, w, GeodesicInterpolation(); kwargs...)
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
mid_point(::Euclidean{Tuple{}}, p1::T, p2::T) where {T<:Number} = (p1 + p2) / 2

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
LinearAlgebra.norm(::MetricManifold{ℝ,<:Manifold,EuclideanMetric}, p, X) = norm(X)

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

@doc raw"""
    project(M::Euclidean, p)

Project an arbitrary point `p` onto the [`Euclidean`](@ref) manifold `M`, which
is of course just the identity map.
"""
project(::Euclidean, ::Any)
project(::Euclidean{Tuple{}}, p::T) where {T<:Number} = p

project!(::Euclidean, q, p) = copyto!(q, p)

"""
    project(M::Euclidean, p, X)

Project an arbitrary vector `X` into the tangent space of a point `p` on the
[`Euclidean`](@ref) `M`, which is just the identity, since any tangent
space of `M` can be identified with all of `M`.
"""
project(::Euclidean, ::Any, ::Any)
project(::Euclidean{Tuple{}}, ::T, X::T) where {T<:Number} = X

project!(::Euclidean, Y, p, X) = copyto!(Y, X)

"""
    representation_size(M::Euclidean)

Return the array dimensions required to represent an element on the
[`Euclidean`](@ref) `M`, i.e. the vector of all array dimensions.
"""
@generated representation_size(::Euclidean{N}) where {N} = size_to_tuple(N)
@generated representation_size(::Euclidean{Tuple{}}) = ()

function retract(M::Euclidean{Tuple{}}, p::T, q::T) where {T<:Number}
    return retract(M, p, q, ExponentialRetraction())
end
function retract(
    M::Euclidean{Tuple{}},
    p::T,
    q::T,
    ::ExponentialRetraction,
) where {T<:Number}
    return exp(M, p, q)
end

"""
    sharp(M::Euclidean, p, ξ)

Transform the cotangent vector `ξ` at `p` on the [`Euclidean`](@ref) `M` to a tangent vector `X`.
Since cotangent and tangent vectors can directly be identified in the [`Euclidean`](@ref)
case, this yields just the identity.
"""
sharp(::Euclidean, ::Any...)

function sharp(::Euclidean{Tuple{}}, ::T, ξ::CoTFVector) where {T<:Number}
    return FVector(TangentSpace, ξ.data)
end

sharp!(::Euclidean, X::TFVector, p, ξ::CoTFVector) = copyto!(X, ξ)

function Base.show(io::IO, ::Euclidean{N,𝔽}) where {N,𝔽}
    return print(io, "Euclidean($(join(N.parameters, ", ")); field = $(𝔽))")
end

function vector_transport_direction(
    M::Euclidean{Tuple{}},
    p::T,
    X::T,
    Y::T,
    m::AbstractVectorTransportMethod,
) where {T<:Number}
    q = exp(M, p, Y)
    return vector_transport_to(M, p, X, q, m)
end

"""
    vector_transport_to(M::Euclidean, p, X, q, ::AbstractVectorTransportMethod)

Transport the vector `X` from the tangent space at `p` to the tangent space at `q`
on the [`Euclidean`](@ref) `M`, which simplifies to the identity.
"""
vector_transport_to(::Euclidean, ::Any, ::Any, ::Any, ::AbstractVectorTransportMethod)
function vector_transport_to(
    ::Euclidean{Tuple{}},
    ::T,
    X::T,
    ::T,
    ::AbstractVectorTransportMethod,
) where {T<:Number}
    return X
end

function vector_transport_to!(
    ::Euclidean,
    Y,
    ::Any,
    X,
    ::Any,
    ::AbstractVectorTransportMethod,
)
    return copyto!(Y, X)
end

for VT in ManifoldsBase.VECTOR_TRANSPORT_DISAMBIGUATION
    eval(
        quote
            @invoke_maker 6 AbstractVectorTransportMethod vector_transport_to!(
                M::Euclidean,
                Y,
                p,
                X,
                q,
                B::$VT,
            )
        end,
    )
end

Statistics.var(::Euclidean, x::AbstractVector; kwargs...) = sum(var(x; kwargs...))
function Statistics.var(::Euclidean, x::AbstractVector{T}, m::T; kwargs...) where {T}
    return sum(var(x; mean=m, kwargs...))
end

"""
    zero_tangent_vector(M::Euclidean, x)

Return the zero vector in the tangent space of `x` on the [`Euclidean`](@ref)
`M`, which here is just a zero filled array the same size as `x`.
"""
zero_tangent_vector(::Euclidean, ::Any...)
zero_tangent_vector(::Euclidean{Tuple{}}, p::T) where {T<:Number} = zero(p)

zero_tangent_vector!(::Euclidean, v, ::Any) = fill!(v, 0)
