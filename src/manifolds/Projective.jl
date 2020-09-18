abstract type AbstractProjective{𝔽} <:
              AbstractEmbeddedManifold{𝔽,DefaultIsometricEmbeddingType} end

struct Projective{N,𝔽} <: AbstractProjective{𝔽} end
Projective(n::Int, field::AbstractNumbers = ℝ) = Projective{n,field}()

struct ArrayProjective{N,𝔽} <: AbstractProjective{𝔽} where {N<:Tuple} end
function ArrayProjective(n::Vararg{Int,I}; field::AbstractNumbers = ℝ) where {I}
    return ArrayProjective{Tuple{n...},field}()
end

function check_manifold_point(M::AbstractProjective, p; kwargs...)
    mpv = invoke(
        check_manifold_point,
        Tuple{(typeof(get_embedding(M))),typeof(p)},
        get_embedding(M),
        p;
        kwargs...,
    )
    mpv === nothing || return mpv
    if !isapprox(norm(p), 1; kwargs...)
        return DomainError(
            norm(p),
            "The point $(p) does not lie on the $(M) since its norm is not 1.",
        )
    end
    return nothing
end

function check_tangent_vector(
    M::AbstractProjective,
    p,
    X;
    check_base_point = true,
    kwargs...,
)
    if check_base_point
        mpe = check_manifold_point(M, p; kwargs...)
        mpe === nothing || return mpe
    end
    mpv = invoke(
        check_tangent_vector,
        Tuple{typeof(get_embedding(M)),typeof(p),typeof(X)},
        get_embedding(M),
        p,
        X;
        check_base_point = false, # already checked above
        kwargs...,
    )
    mpv === nothing || return mpv
    if !isapprox(real(dot(p, X)), 0; kwargs...)
        return DomainError(
            dot(p, X),
            "The vector $(X) is not a tangent vector to $(p) on $(M), since it is not orthogonal in the embedding.",
        )
    end
    return nothing
end

function decorated_manifold(M::AbstractProjective{𝔽}) where {𝔽}
    return Euclidean(representation_size(M)...; field = 𝔽)
end
get_embedding(M::AbstractProjective{𝔽}) where {𝔽} = decorated_manifold(M)

distance(::AbstractProjective, p, q) = acos(min(abs(dot(p, q)), 1))

embed!(::AbstractProjective, q, p) = (q .= p)

embed!(::AbstractProjective, Y, p, X) = (Y .= X)

function exp!(M::AbstractProjective, q, p, X)
    θ = norm(M, p, X)
    q .= cos(θ) .* p .+ usinc(θ) .* X
    return q
end

flat!(::AbstractProjective, ξ::CoTFVector, p, X::TFVector) = copyto!(ξ, X)

injectivity_radius(::AbstractProjective) = π / 2
injectivity_radius(::AbstractProjective, ::ExponentialRetraction) = π / 2
injectivity_radius(::AbstractProjective, ::Any) = π / 2
injectivity_radius(::AbstractProjective, ::Any, ::ExponentialRetraction) = π / 2
eval(
    quote
        @invoke_maker 1 Manifold injectivity_radius(
            M::AbstractProjective,
            rm::AbstractRetractionMethod,
        )
    end,
)

function inverse_retract!(
    ::AbstractProjective,
    X,
    p,
    q,
    ::Union{ProjectionInverseRetraction,PolarInverseRetraction,QRInverseRetraction},
)
    return (X .= q ./ dot(p, q) .- p)
end

function Base.isapprox(M::AbstractProjective, p, X, Y; kwargs...)
    return isapprox(sqrt(inner(M, p, zero_tangent_vector(M, p), X - Y)), 0; kwargs...)
end
function Base.isapprox(M::AbstractProjective, p, q; kwargs...)
    return isapprox(distance(M, p, q), 0; kwargs...)
end

function log!(M::AbstractProjective, X, p, q)
    z = dot(q, p)
    absz = min(abs(z), 1)
    θ = acos(absz)
    signz = z isa Real ? sign(z) : z / ifelse(iszero(absz), one(absz), absz)
    X .= (signz .* q .- absz .* p) ./ usinc(θ)
    return project!(M, X, p, X)
end

@doc raw"""
    manifold_dimension(M::AbstractProjective)

Return the dimension of the [`AbstractProjective`](@ref) `M`, respectively i.e. the
dimension of the embedding -1.
"""
manifold_dimension(M::AbstractProjective) = manifold_dimension(get_embedding(M)) - 1

"""
    mean(
        M::AbstractProjective,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = GeodesicInterpolationWithinRadius(π/4);
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(M::Manifold, args...)) of `x` using
[`GeodesicInterpolationWithinRadius`](@ref).
"""
mean(::AbstractProjective, ::Any...)

function Statistics.mean!(
    M::AbstractProjective,
    p,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
)
    return mean!(M, p, x, w, GeodesicInterpolationWithinRadius(π / 4); kwargs...)
end

function mid_point!(::Projective, q, p1, p2)
    z = dot(p1, p2)
    absz = abs(z)
    signz = z isa Real ? sign(z) : z / ifelse(iszero(absz), one(absz), absz)
    q .= (p1 .+ signz' .* p2) ./ sqrt(2 + 2absz)
    return q
end

"""
    normal_tvector_distribution(M::Projective{n,ℝ}, p, σ)

Generate a distribution in the tangent space at `p` by generating a
normal distribution in ambient space with standard deviation `σ`
projected to the tangent space at `p`.
"""
function normal_tvector_distribution(M::Projective{n,ℝ}, p, σ) where {n}
    d = Distributions.MvNormal(zero(p), σ)
    return ProjectedFVectorDistribution(TangentBundleFibers(M), p, d, project!, p)
end

project!(::AbstractProjective, q, p) = copyto!(q, p ./ norm(p))

project!(::AbstractProjective, Y, p, X) = (Y .= X .- dot(p, X) .* p)

@doc raw"""
    representation_size(M::AbstractProjective)

Return the size points on the [`AbstractProjective`](@ref) `M` are represented as, i.e., the
representation size of the embedding.
"""
@generated representation_size(::ArrayProjective{N}) where {N} = size_to_tuple(N)
@generated representation_size(::Projective{N}) where {N} = (N + 1,)

function retract!(
    M::AbstractProjective,
    q,
    p,
    X,
    ::Union{ProjectionRetraction,PolarRetraction,QRRetraction},
)
    q .= p .+ X
    return project!(M, q, q)
end

Base.show(io::IO, ::Projective{n,𝔽}) where {n,𝔽} = print(io, "Projective($(n), $(𝔽))")
function Base.show(io::IO, ::ArrayProjective{N,𝔽}) where {N,𝔽}
    return print(io, "ArrayProjective($(join(N.parameters, ", ")); field = $(𝔽))")
end

"""
    uniform_distribution(M::Projective{ℝ}, p)

Uniform distribution on given [`Projective`](@ref) `M`. Generated points will be of
similar type as `p`.
"""
function uniform_distribution(M::Projective{ℝ}, p)
    d = Distributions.MvNormal(zero(p), 1.0)
    return ProjectedPointDistribution(M, d, project!, p)
end
