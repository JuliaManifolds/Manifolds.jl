@doc raw"""
    Euclidean{T,𝔽} <: AbstractManifold{𝔽}

Euclidean vector space.

# Constructor

    Euclidean(n)

Generate the ``n``-dimensional vector space ``ℝ^n``.

    Euclidean(n₁,n₂,...,nᵢ; field=ℝ, parameter::Symbol = :field)
    𝔽^(n₁,n₂,...,nᵢ) = Euclidean(n₁,n₂,...,nᵢ; field=𝔽)

Generate the vector space of ``k = n_1 ⋅ n_2 ⋅ … ⋅ n_i`` values, i.e. the
manifold ``𝔽^{n_1, n_2, …, n_i}``, ``𝔽\in\{ℝ,ℂ\}``, whose
elements are interpreted as ``n_1 × n_2 × … × n_i`` arrays.
For ``i=2`` we obtain a matrix space.
The default `field=ℝ` can also be set to `field=ℂ`.
The dimension of this space is ``k \dim_ℝ 𝔽``, where ``\dim_ℝ 𝔽`` is the
[`real_dimension`](@extref `ManifoldsBase.real_dimension-Tuple{ManifoldsBase.AbstractNumbers}`) of the field ``𝔽``.

`parameter`: whether a type parameter should be used to store `n`. By default size
is stored in type. Value can either be `:field` or `:type`.

    Euclidean(; field=ℝ)

Generate the 1D Euclidean manifold for an `ℝ`-, `ℂ`-valued  real- or complex-valued immutable
values (in contrast to 1-element arrays from the constructor above).
"""
struct Euclidean{𝔽, T} <: AbstractDecoratorManifold{𝔽} where {T}
    size::T
end

function Euclidean(
        n::Vararg{Int, I};
        field::AbstractNumbers = ℝ,
        parameter::Symbol = :type,
    ) where {I}
    size = wrap_type_parameter(parameter, n)
    return Euclidean{field, typeof(size)}(size)
end

function adjoint_Jacobi_field(::Euclidean{𝔽, Tuple{}}, p, q, t, X, β::Tβ) where {𝔽, Tβ}
    return X
end
function adjoint_Jacobi_field(
        ::Euclidean{𝔽, TypeParameter{Tuple{}}},
        p,
        q,
        t,
        X,
        β::Tβ,
    ) where {𝔽, Tβ}
    return X
end

Base.:^(𝔽::AbstractNumbers, n) = Euclidean(n...; field = 𝔽)

Base.:^(M::Euclidean, n::Int) = ^(M, (n,))
function Base.:^(M::Euclidean{𝔽, <:Tuple}, n::NTuple{N, Int}) where {𝔽, N}
    size = get_parameter(M.size)
    return Euclidean(size..., n...; field = 𝔽, parameter = :field)
end
function Base.:^(M::Euclidean{𝔽, <:TypeParameter}, n::NTuple{N, Int}) where {𝔽, N}
    size = get_parameter(M.size)
    return Euclidean(size..., n...; field = 𝔽, parameter = :type)
end

function allocation_promotion_function(
        ::Euclidean{ℂ},
        ::Union{typeof(get_vector), typeof(get_coordinates)},
        ::Tuple,
    )
    return complex
end

function check_point(M::Euclidean{𝔽, N}, p) where {𝔽, N}
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

function check_vector(M::Euclidean{𝔽, N}, p, X; kwargs...) where {𝔽, N}
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

connection(::Euclidean) = LeviCivitaConnection()

default_approximation_method(::Euclidean, ::typeof(mean)) = EfficientEstimator()

function default_approximation_method(::Euclidean, ::typeof(median), ::Type{<:Number})
    return EfficientEstimator()
end

function det_local_metric(
        ::MetricManifold{𝔽, <:AbstractManifold, EuclideanMetric},
        p,
        ::InducedBasis{𝔽, TangentSpaceType, <:RetractionAtlas},
    ) where {𝔽}
    return one(eltype(p))
end

function diagonalizing_projectors(::Euclidean, p, X)
    return ((zero(number_eltype(p)), IdentityProjector()),)
end

"""
    distance(M::Euclidean, p, q, r::Real=2)

Compute the Euclidean distance between two points on the [`Euclidean`](@ref)
manifold `M`, i.e. for vectors it's just the norm of the difference, for matrices
and higher order arrays, the matrix and tensor Frobenius norm, respectively.
Specifying further an `r≠2`, other norms, like the 1-norm or the ∞-norm can also be computed.
"""
Base.@propagate_inbounds function distance(M::Euclidean, p, q)
    # Inspired by euclidean distance calculation in Distances.jl
    # Much faster for large p, q than a naive implementation
    @boundscheck if axes(p) != axes(q)
        throw(DimensionMismatch("At last one of $p and $q does not belong to $M"))
    end
    s = zero(eltype(p))
    @inbounds begin # COV_EXCL_LINE
        @simd for I in eachindex(p, q) # COV_EXCL_LINE
            p_i = p[I]
            q_i = q[I]
            s += abs2(p_i - q_i)
        end # COV_EXCL_LINE
    end # COV_EXCL_LINE
    return sqrt(s)
end
distance(M::Euclidean, p, q, r::Real) = norm(p - q, r)
distance(::Euclidean{𝔽, TypeParameter{Tuple{1}}}, p::Number, q::Number) where {𝔽} = abs(p - q)
distance(::Euclidean{𝔽, TypeParameter{Tuple{}}}, p::Number, q::Number) where {𝔽} = abs(p - q)
distance(::Euclidean{𝔽, Tuple{Int}}, p::Number, q::Number) where {𝔽} = abs(p - q) # for 1-dimensional Euclidean
distance(::Euclidean{𝔽, Tuple{}}, p::Number, q::Number) where {𝔽} = abs(p - q)

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
        ::EmbeddedManifold{𝔽, Euclidean{𝔽, nL}, Euclidean{𝔽2, mL}},
        q,
        p,
    ) where {𝔽, 𝔽2, nL, mL}
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
Base.exp(::Euclidean, p, X) = p + X
exp_fused(::Euclidean, p, X, t::Number) = p .+ t .* X

exp!(::Euclidean, q, p, X) = (q .= p .+ X)
exp_fused!(::Euclidean, q, p, X, t::Number) = (q .= p .+ t .* X)
exp_fused!(::Euclidean{𝔽, TypeParameter{Tuple{}}}, q, p, X, t::Number) where {𝔽} = (q .= p[] + t * X[])
exp_fused!(::Euclidean{𝔽, Tuple{}}, q, p, X, t::Number) where {𝔽} = (q .= p[] + t * X[])

function get_basis_diagonalizing(
        M::Euclidean,
        p,
        B::DiagonalizingOrthonormalBasis{𝔽},
    ) where {𝔽}
    vecs = get_vectors(M, p, get_basis(M, p, DefaultOrthonormalBasis(𝔽)))
    eigenvalues = zeros(real(eltype(p)), manifold_dimension(M))
    return CachedBasis(B, DiagonalizingBasisData(B.frame_direction, eigenvalues, vecs))
end

function get_coordinates_orthonormal(::Euclidean{ℝ}, p, X, ::RealNumbers)
    return vec(X)
end
function get_coordinates_orthonormal(::Euclidean{ℂ}, p, X, ::ComplexNumbers)
    return vec(X)
end

function get_coordinates_orthonormal!(::Euclidean{ℝ}, c, p, X, ::RealNumbers)
    copyto!(c, vec(X))
    return c
end
function get_coordinates_orthonormal!(::Euclidean{ℂ}, c, p, X, ::ComplexNumbers)
    copyto!(c, vec(X))
    return c
end

function get_coordinates_induced_basis!(
        M::Euclidean,
        c,
        p,
        X,
        ::InducedBasis{ℝ, TangentSpaceType, <:RetractionAtlas},
    )
    copyto!(c, vec(X))
    return c
end

function get_coordinates_orthonormal!(::Euclidean{ℂ}, c, ::Any, X, ::RealNumbers)
    Xvec = vec(X)
    d = div(length(c), 2)
    view(c, 1:d) .= real.(Xvec)
    view(c, (d + 1):(2d)) .= imag.(Xvec)
    return c
end

function get_coordinates_diagonalizing!(
        ::Euclidean{ℂ},
        c,
        ::Any,
        X,
        ::DiagonalizingOrthonormalBasis{ℝ},
    )
    Xvec = vec(X)
    d = div(length(c), 2)
    view(c, 1:d) .= real.(Xvec)
    view(c, (d + 1):(2d)) .= imag.(Xvec)
    return c
end
function get_coordinates_diagonalizing!(
        ::Euclidean{𝔽},
        c,
        p,
        X,
        ::DiagonalizingOrthonormalBasis{𝔽},
    ) where {𝔽}
    copyto!(c, vec(X))
    return c
end

function get_vector_orthonormal(M::Euclidean{ℝ}, ::Any, c, ::RealNumbers)
    S = representation_size(M)
    return reshape(c, S)
end
function get_vector_orthonormal(M::Euclidean{ℂ}, ::Any, c, ::ComplexNumbers)
    S = representation_size(M)
    return reshape(c, S)
end
function get_vector_orthonormal(
        ::Euclidean{ℝ, TypeParameter{Tuple{N}}},
        ::Any,
        c,
        ::RealNumbers,
    ) where {N}
    # this method is defined just to skip a reshape
    return c
end
function get_vector_orthonormal(::Euclidean{ℝ, Tuple{Int}}, ::Any, c, ::RealNumbers)
    # this method is defined just to skip a reshape
    return c
end
function get_vector_orthonormal(
        ::Euclidean{ℝ, <:TypeParameter},
        ::SArray{S},
        c,
        ::RealNumbers,
    ) where {S}
    return SArray{S}(c)
end
function get_vector_orthonormal(
        ::Euclidean{ℝ, TypeParameter{Tuple{N}}},
        ::SArray{S},
        c,
        ::RealNumbers,
    ) where {N, S}
    # probably doesn't need rewrapping in SArray
    return c
end
function get_vector_orthonormal(
        ::Euclidean{ℝ, TypeParameter{Tuple{N}}},
        ::SizedArray{S},
        c,
        ::RealNumbers,
    ) where {N, S}
    # probably doesn't need rewrapping in SizedArray
    return c
end

function get_vector_orthonormal!(
        ::Euclidean{ℝ, TypeParameter{Tuple{N}}},
        Y,
        ::Any,
        c,
        ::RealNumbers,
    ) where {N}
    # this method is defined just to skip a reshape
    copyto!(Y, c)
    return Y
end
function get_vector_orthonormal!(M::Euclidean{ℝ}, Y, ::Any, c, ::RealNumbers)
    S = representation_size(M)
    copyto!(Y, reshape(c, S))
    return Y
end
function get_vector_orthonormal!(M::Euclidean{ℂ}, Y, ::Any, c, ::ComplexNumbers)
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
function get_vector_orthonormal!(M::Euclidean{ℂ}, Y, ::Any, c, ::RealNumbers)
    S = representation_size(M)
    N = div(length(c), 2)
    copyto!(Y, reshape(c[1:N] .+ im .* c[(N + 1):end], S))
    return Y
end
function get_vector_diagonalizing!(
        M::Euclidean{ℂ},
        Y,
        ::Any,
        c,
        ::DiagonalizingOrthonormalBasis{ℝ},
    )
    S = representation_size(M)
    N = div(length(c), 2)
    copyto!(Y, reshape(c[1:N] .+ im .* c[(N + 1):end], S))
    return Y
end

has_components(::Euclidean) = true

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

``i ≤ j ≤ i`` it holds ``1 ≤ k_j ≤ n_j`` and ``\overline{⋅}`` denotes the complex conjugate.

For the special case of ``i ≤ 2``, i.e. matrices and vectors, this simplifies to

````math
g_p(X,Y) = \operatorname{tr}(X^{\mathrm{H}}Y),
````

where ``⋅^{\mathrm{H}}`` denotes the Hermitian, i.e. complex conjugate transposed.
"""
inner(::Euclidean, ::Any...)
@inline inner(::Euclidean, p, X, Y) = dot(X, Y)
@inline function inner(
        ::MetricManifold{𝔽, <:AbstractManifold, EuclideanMetric},
        p,
        X,
        Y,
    ) where {𝔽}
    return dot(X, Y)
end

function inverse_local_metric(
        M::MetricManifold{𝔽, <:AbstractManifold, EuclideanMetric},
        p,
        B::InducedBasis{𝔽, TangentSpaceType, <:RetractionAtlas},
    ) where {𝔽}
    return local_metric(M, p, B)
end
function inverse_local_metric(
        M::Euclidean,
        p,
        B::InducedBasis{𝔽, TangentSpaceType, <:RetractionAtlas},
    ) where {𝔽}
    return local_metric(M, p, B)
end

"""
    is_flat(::Euclidean)

Return true. [`Euclidean`](@ref) is a flat manifold.
"""
is_flat(M::Euclidean) = true

function jacobi_field(::Euclidean{𝔽, TypeParameter{Tuple{}}}, p, q, t, X, β::Tβ) where {𝔽, Tβ}
    return X
end
function jacobi_field(::Euclidean{𝔽, Tuple{}}, p, q, t, X, β::Tβ) where {𝔽, Tβ}
    return X
end

function local_metric(
        ::MetricManifold{𝔽, <:AbstractManifold, EuclideanMetric},
        p,
        B::InducedBasis{𝔽, TangentSpaceType, <:RetractionAtlas},
    ) where {𝔽}
    return Diagonal(ones(SVector{size(p, 1), eltype(p)}))
end
function local_metric(
        ::Euclidean,
        p,
        B::InducedBasis{𝔽, TangentSpaceType, <:RetractionAtlas},
    ) where {𝔽}
    return Diagonal(ones(SVector{size(p, 1), eltype(p)}))
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
Base.log(::Euclidean{𝔽, TypeParameter{Tuple{}}}, p::Number, q::Number) where {𝔽} = q - p
Base.log(::Euclidean, p, q) = q .- p

log!(::Euclidean, X, p, q) = (X .= q .- p)

function log_local_metric_density(
        ::MetricManifold{𝔽, <:AbstractManifold, EuclideanMetric},
        p,
        ::InducedBasis{𝔽, TangentSpaceType, <:RetractionAtlas},
    ) where {𝔽}
    return zero(eltype(p))
end

_product_of_dimensions(M::Euclidean) = prod(get_parameter(M.size))

"""
    manifold_dimension(M::Euclidean)

Return the manifold dimension of the [`Euclidean`](@ref) `M`, i.e.
the product of all array dimensions and the [`real_dimension`](@extref `ManifoldsBase.real_dimension-Tuple{ManifoldsBase.AbstractNumbers}`) of the
underlying number system.
"""
function manifold_dimension(M::Euclidean{𝔽}) where {𝔽}
    return _product_of_dimensions(M) * real_dimension(𝔽)
end
manifold_dimension(::Euclidean{𝔽, TypeParameter{Tuple{}}}) where {𝔽} = real_dimension(𝔽)

"""
    manifold_volume(::Euclidean)

Return volume of the [`Euclidean`](@ref) manifold, i.e. infinity.
"""
manifold_volume(::Euclidean) = Inf

function Statistics.mean(
        ::Union{Euclidean{𝔽, TypeParameter{Tuple{}}}, Euclidean{𝔽, Tuple{}}},
        x::AbstractVector,
        ::EfficientEstimator;
        kwargs...,
    ) where {𝔽}
    return mean(x)
end
function Statistics.mean(
        ::Union{Euclidean{𝔽, TypeParameter{Tuple{}}}, Euclidean{𝔽, Tuple{}}},
        x::AbstractVector,
        w::AbstractWeights,
        ::EfficientEstimator;
        kwargs...,
    ) where {𝔽}
    return mean(x, w)
end
#
# When Statistics / Statsbase.mean! is consistent with mean, we can pass this on to them as well
function Statistics.mean!(
        ::Euclidean,
        y,
        x::AbstractVector,
        ::EfficientEstimator;
        kwargs...,
    )
    n = length(x)
    copyto!(y, first(x))
    @inbounds for j in 2:n
        y .+= x[j]
    end
    y ./= n
    return y
end
function Statistics.mean!(
        ::Euclidean,
        y,
        x::AbstractVector,
        w::AbstractWeights,
        ::EfficientEstimator;
        kwargs...,
    )
    n = length(x)
    if length(w) != n
        throw(
            DimensionMismatch(
                "The number of weights ($(length(w))) does not match the number of points for the mean ($(n)).",
            ),
        )
    end
    copyto!(y, first(x))
    y .*= first(w)
    @inbounds for j in 2:n
        iszero(w[j]) && continue
        y .+= w[j] .* x[j]
    end
    y ./= sum(w)
    return y
end

function StatsBase.mean_and_var(
        ::Union{Euclidean{𝔽, TypeParameter{Tuple{}}}, Euclidean{𝔽, Tuple{}}},
        x::AbstractVector{<:Number};
        kwargs...,
    ) where {𝔽}
    m, v = mean_and_var(x; kwargs...)
    return m, sum(v)
end
function StatsBase.mean_and_var(
        ::Union{Euclidean{𝔽, TypeParameter{Tuple{}}}, Euclidean{𝔽, Tuple{}}},
        x::AbstractVector{<:Number},
        w::AbstractWeights;
        corrected = false,
        kwargs...,
    ) where {𝔽}
    m, v = mean_and_var(x, w; corrected = corrected, kwargs...)
    return m, sum(v)
end

function Statistics.median(
        ::Union{Euclidean{𝔽, TypeParameter{Tuple{}}}, Euclidean{𝔽, Tuple{}}},
        x::AbstractVector{<:Number},
        ::EfficientEstimator;
        kwargs...,
    ) where {𝔽}
    return median(x)
end
function Statistics.median(
        ::Union{Euclidean{𝔽, TypeParameter{Tuple{}}}, Euclidean{𝔽, Tuple{}}},
        x::AbstractVector{<:Number},
        w::AbstractWeights,
        ::EfficientEstimator;
        kwargs...,
    ) where {𝔽}
    return median(x, w)
end

metric(::Euclidean) = EuclideanMetric()

mid_point(::Euclidean, p1, p2) = (p1 .+ p2) ./ 2
function mid_point(
        ::Union{Euclidean{𝔽, TypeParameter{Tuple{}}}, Euclidean{𝔽, Tuple{}}},
        p1::Number,
        p2::Number,
    ) where {𝔽}
    return (p1 + p2) / 2
end

function mid_point!(::Euclidean, q, p1, p2)
    q .= (p1 .+ p2) ./ 2
    return q
end

@doc raw"""
    norm(M::Euclidean, p, X, r::Real=2)

Compute the norm of a tangent vector `X` at `p` on the [`Euclidean`](@ref)
`M`, i.e. since every tangent space can be identified with `M` itself
in this case, just the (Frobenius) norm of `X`. Specifying `r`, other norms are available as well
"""
LinearAlgebra.norm(::Euclidean, ::Any, X, r::Real = 2) = norm(X, r)
function LinearAlgebra.norm(
        ::MetricManifold{ℝ, <:AbstractManifold, EuclideanMetric},
        p,
        X,
        r::Real = 2,
    )
    return norm(X, r)
end

function project!(
        ::EmbeddedManifold{𝔽, Euclidean{𝔽, nL}, Euclidean{𝔽2, mL}},
        q,
        p,
    ) where {nL, mL, 𝔽, 𝔽2}
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
    parallel_transport_direction(M::Euclidean, p, X, d)

the parallel transport on [`Euclidean`](@ref) is the identity, i.e. returns `X`.
"""
parallel_transport_direction(::Euclidean, ::Any, X, ::Any) = X
parallel_transport_direction!(::Euclidean, Y, ::Any, X, ::Any) = copyto!(Y, X)

"""
    parallel_transport_to(M::Euclidean, p, X, q)

the parallel transport on [`Euclidean`](@ref) is the identity, i.e. returns `X`.
"""
parallel_transport_to(::Euclidean, ::Any, X, ::Any) = X
parallel_transport_to!(::Euclidean, Y, ::Any, X, ::Any) = copyto!(Y, X)
function parallel_transport_to!(::Euclidean{𝔽, TypeParameter{Tuple{}}}, Y, ::Any, X, ::Any) where {𝔽}
    return copyto!(Y, X[])
end
parallel_transport_to!(::Euclidean{𝔽, Tuple{}}, Y, ::Any, X, ::Any) where {𝔽} = copyto!(Y, X[])

@doc raw"""
    project(M::Euclidean, p)

Project an arbitrary point `p` onto the [`Euclidean`](@ref) manifold `M`, which
is of course just the identity map.
"""
project(::Euclidean, ::Any)
project(::Euclidean{𝔽, TypeParameter{Tuple{}}}, p::Number) where {𝔽} = p
project(::Euclidean{𝔽, Tuple{}}, p::Number) where {𝔽} = p

project!(::Euclidean, q, p) = copyto!(q, p)

"""
    project(M::Euclidean, p, X)

Project an arbitrary vector `X` into the tangent space of a point `p` on the
[`Euclidean`](@ref) `M`, which is just the identity, since any tangent
space of `M` can be identified with all of `M`.
"""
project(::Euclidean, ::Any, ::Any)
project(::Euclidean{𝔽, TypeParameter{Tuple{}}}, ::Number, X::Number) where {𝔽} = X
project(::Euclidean{𝔽, Tuple{}}, ::Number, X::Number) where {𝔽} = X

project!(::Euclidean, Y, p, X) = copyto!(Y, X)

function Random.rand!(
        rng::AbstractRNG, ::Euclidean, pX;
        σ = one(eltype(pX)),
        vector_at = nothing,
    )
    pX .= randn(rng, eltype(pX), size(pX)) .* σ
    return pX
end

"""
    representation_size(M::Euclidean)

Return the array dimensions required to represent an element on the
[`Euclidean`](@ref) `M`, i.e. the vector of all array dimensions.
"""
representation_size(M::Euclidean) = get_parameter(M.size)

function retract(M::Euclidean{𝔽, TypeParameter{Tuple{}}}, p::Number, q::Number) where {𝔽}
    return retract(M, p, q, ExponentialRetraction())
end
function retract(
        M::Euclidean{𝔽, TypeParameter{Tuple{}}},
        p::Number, q::Number, ::ExponentialRetraction,
    ) where {𝔽}
    return exp(M, p, q)
end

@doc raw"""
    riemann_tensor(M::Euclidean, p, X, Y, Z)

Compute the Riemann tensor ``R(X,Y)Z`` at point `p` on [`Euclidean`](@ref) manifold `M`.
Its value is always the zero tangent vector.
"""
riemann_tensor(M::Euclidean, p, X, Y, Z)

function riemann_tensor!(::Euclidean, Xresult, p, X, Y, Z)
    return fill!(Xresult, 0)
end

@doc raw"""
    sectional_curvature(::Euclidean, p, X, Y)

Sectional curvature of [`Euclidean`](@ref) manifold `M` is 0.
"""
function sectional_curvature(::Euclidean, p, X, Y)
    return 0.0
end

@doc raw"""
    sectional_curvature_max(::Euclidean)

Sectional curvature of [`Euclidean`](@ref) manifold `M` is 0.
"""
function sectional_curvature_max(::Euclidean)
    return 0.0
end

@doc raw"""
    sectional_curvature_min(M::Euclidean)

Sectional curvature of [`Euclidean`](@ref) manifold `M` is 0.
"""
function sectional_curvature_min(::Euclidean)
    return 0.0
end

function Base.show(io::IO, M::Euclidean{𝔽, N}) where {N <: Tuple, 𝔽}
    size = get_parameter(M.size)
    return print(io, "Euclidean($(join(size, ", ")); field=$(𝔽), parameter=:field)")
end
function Base.show(io::IO, M::Euclidean{𝔽, N}) where {N <: TypeParameter, 𝔽}
    size = get_parameter(M.size)
    return print(io, "Euclidean($(join(size, ", ")); field=$(𝔽))")
end
#
# Vector Transport
#
# The following functions are defined on layer 1 already, since
# a) its independent of the transport or retraction method
# b) no ambiguities occur
# c) Euclidean is so basic, that these are plain defaults
#
function vector_transport_direction(
        M::Euclidean,
        p,
        X,
        ::Any,
        ::AbstractVectorTransportMethod = default_vector_transport_method(M, typeof(p)),
        ::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
    )
    return X
end
function vector_transport_direction!(
        M::Euclidean,
        Y,
        p,
        X,
        ::Any,
        ::AbstractVectorTransportMethod = default_vector_transport_method(M, typeof(p)),
        ::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
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
        p,
        X,
        ::Any,
        ::AbstractVectorTransportMethod = default_vector_transport_method(M, typeof(p)),
        ::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
    )
    return X
end

function vector_transport_to!(
        M::Euclidean,
        Y,
        p,
        X,
        ::Any,
        ::AbstractVectorTransportMethod = default_vector_transport_method(M, typeof(p)),
        ::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
    )
    return copyto!(Y, X)
end

Statistics.var(::Euclidean, x::AbstractVector; kwargs...) = sum(var(x; kwargs...))
function Statistics.var(::Euclidean, x::AbstractVector{<:Number}, m::Number; kwargs...)
    return sum(var(x; mean = m, kwargs...))
end

@doc raw"""
    volume_density(M::Euclidean, p, X)

Return volume density function of [`Euclidean`](@ref) manifold `M`, i.e. 1.
"""
function volume_density(::Euclidean, p, X)
    return one(eltype(X))
end

@doc raw"""
    Y = Weingarten(M::Euclidean, p, X, V)
    Weingarten!(M::Euclidean, Y, p, X, V)

Compute the Weingarten map ``\mathcal W_p`` at `p` on the [`Euclidean`](@ref) `M` with respect to the
tangent vector ``X \in T_p\mathcal M`` and the normal vector ``V \in N_p\mathcal M``.

Since this a flat space by itself, the result is always the zero tangent vector.
"""
Weingarten(::Euclidean, p, X, V)

Weingarten!(::Euclidean, Y, p, X, V) = fill!(Y, 0)

"""
    zero_vector(M::Euclidean, p)

Return the zero vector in the tangent space of `p` on the [`Euclidean`](@ref)
`M`, which here is just a zero filled array the same size as `p`.
"""
zero_vector(::Euclidean, ::Any...)
zero_vector(::Euclidean{𝔽, TypeParameter{Tuple{}}}, p::Number) where {𝔽} = zero(p)
zero_vector(::Euclidean{𝔽, Tuple{}}, p::Number) where {𝔽} = zero(p)

zero_vector!(::Euclidean, X, ::Any) = fill!(X, 0)
