abstract type AbstractProjectiveSpace{𝔽} <:
              AbstractEmbeddedManifold{𝔽,DefaultIsometricEmbeddingType} end

struct ProjectiveSpace{N,𝔽} <: AbstractProjectiveSpace{𝔽} end
ProjectiveSpace(n::Int, field::AbstractNumbers = ℝ) = ProjectiveSpace{n,field}()

struct ArrayProjectiveSpace{N,𝔽} <: AbstractProjectiveSpace{𝔽} where {N<:Tuple} end
function ArrayProjectiveSpace(n::Vararg{Int,I}; field::AbstractNumbers = ℝ) where {I}
    return ArrayProjectiveSpace{Tuple{n...},field}()
end

function allocation_promotion_function(::AbstractProjectiveSpace{ℂ}, f, args::Tuple)
    return complex
end

function check_manifold_point(M::AbstractProjectiveSpace, p; kwargs...)
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
    M::AbstractProjectiveSpace,
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

function decorated_manifold(M::AbstractProjectiveSpace{𝔽}) where {𝔽}
    return Euclidean(representation_size(M)...; field = 𝔽)
end
get_embedding(M::AbstractProjectiveSpace{𝔽}) where {𝔽} = decorated_manifold(M)

distance(::AbstractProjectiveSpace, p, q) = acos(min(abs(dot(p, q)), 1))

embed!(::AbstractProjectiveSpace, q, p) = (q .= p)

embed!(::AbstractProjectiveSpace, Y, p, X) = (Y .= X)

function exp!(M::AbstractProjectiveSpace, q, p, X)
    θ = norm(M, p, X)
    q .= cos(θ) .* p .+ usinc(θ) .* X
    return q
end

flat!(::AbstractProjectiveSpace, ξ::CoTFVector, p, X::TFVector) = copyto!(ξ, X)

injectivity_radius(::AbstractProjectiveSpace) = π / 2
injectivity_radius(::AbstractProjectiveSpace, ::ExponentialRetraction) = π / 2
injectivity_radius(::AbstractProjectiveSpace, ::Any) = π / 2
injectivity_radius(::AbstractProjectiveSpace, ::Any, ::ExponentialRetraction) = π / 2
eval(
    quote
        @invoke_maker 1 Manifold injectivity_radius(
            M::AbstractProjectiveSpace,
            rm::AbstractRetractionMethod,
        )
    end,
)

function inverse_retract!(
    ::AbstractProjectiveSpace,
    X,
    p,
    q,
    ::Union{ProjectionInverseRetraction,PolarInverseRetraction,QRInverseRetraction},
)
    return (X .= q ./ dot(p, q) .- p)
end

function Base.isapprox(M::AbstractProjectiveSpace, p, X, Y; kwargs...)
    return isapprox(sqrt(inner(M, p, zero_tangent_vector(M, p), X - Y)), 0; kwargs...)
end
function Base.isapprox(M::AbstractProjectiveSpace, p, q; kwargs...)
    return isapprox(distance(M, p, q), 0; kwargs...)
end

function log!(M::AbstractProjectiveSpace, X, p, q)
    z = dot(p, q)
    cosθ = abs(z)
    X .= (sign(z)' .* q .- cosθ .* p) ./ usinc_from_cos(cosθ)
    return project!(M, X, p, X)
end

@doc raw"""
    manifold_dimension(M::AbstractProjectiveSpace)

Return the dimension of the [`AbstractProjectiveSpace`](@ref) `M`, respectively i.e. the
dimension of the embedding -1.
"""
manifold_dimension(M::AbstractProjectiveSpace) = manifold_dimension(get_embedding(M)) - 1

"""
    mean(
        M::AbstractProjectiveSpace,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = GeodesicInterpolationWithinRadius(π/4);
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(M::Manifold, args...)) of `x` using
[`GeodesicInterpolationWithinRadius`](@ref).
"""
mean(::AbstractProjectiveSpace, ::Any...)

function Statistics.mean!(
    M::AbstractProjectiveSpace,
    p,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
)
    return mean!(M, p, x, w, GeodesicInterpolationWithinRadius(π / 4); kwargs...)
end

function mid_point!(::ProjectiveSpace, q, p1, p2)
    z = dot(p1, p2)
    absz = abs(z)
    signz = z isa Real ? sign(z) : z / ifelse(iszero(absz), one(absz), absz)
    q .= (p1 .+ signz' .* p2) ./ sqrt(2 + 2absz)
    return q
end

"""
    normal_tvector_distribution(M::ProjectiveSpace{n,ℝ}, p, σ)

Generate a distribution in the tangent space at `p` by generating a
normal distribution in ambient space with standard deviation `σ`
projected to the tangent space at `p`.
"""
function normal_tvector_distribution(M::ProjectiveSpace{n,ℝ}, p, σ) where {n}
    d = Distributions.MvNormal(zero(p), σ)
    return ProjectedFVectorDistribution(TangentBundleFibers(M), p, d, project!, p)
end

project!(::AbstractProjectiveSpace, q, p) = copyto!(q, p ./ norm(p))

project!(::AbstractProjectiveSpace, Y, p, X) = (Y .= X .- dot(p, X) .* p)

@doc raw"""
    representation_size(M::AbstractProjectiveSpace)

Return the size points on the [`AbstractProjectiveSpace`](@ref) `M` are represented as, i.e., the
representation size of the embedding.
"""
@generated representation_size(::ArrayProjectiveSpace{N}) where {N} = size_to_tuple(N)
@generated representation_size(::ProjectiveSpace{N}) where {N} = (N + 1,)

function retract!(
    M::AbstractProjectiveSpace,
    q,
    p,
    X,
    ::Union{ProjectionRetraction,PolarRetraction,QRRetraction},
)
    q .= p .+ X
    return project!(M, q, q)
end

function Base.show(io::IO, ::ProjectiveSpace{n,𝔽}) where {n,𝔽}
    return print(io, "ProjectiveSpace($(n), $(𝔽))")
end
function Base.show(io::IO, ::ArrayProjectiveSpace{N,𝔽}) where {N,𝔽}
    return print(io, "ArrayProjectiveSpace($(join(N.parameters, ", ")); field = $(𝔽))")
end

"""
    uniform_distribution(M::ProjectiveSpace{n,ℝ}, p) where {n}

Uniform distribution on given [`ProjectiveSpace`](@ref) `M`. Generated points will be of
similar type as `p`.
"""
function uniform_distribution(M::ProjectiveSpace{n,ℝ}, p) where {n}
    d = Distributions.MvNormal(zero(p), 1.0)
    return ProjectedPointDistribution(M, d, project!, p)
end
