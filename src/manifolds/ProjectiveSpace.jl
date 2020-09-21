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
    if !isapprox(dot(p, X), 0; kwargs...)
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

get_embedding(M::AbstractProjectiveSpace) = decorated_manifold(M)

distance(::AbstractProjectiveSpace, p, q) = acos(min(abs(dot(p, q)), 1))

embed!(::AbstractProjectiveSpace, q, p) = copyto!(q, p)

embed!(::AbstractProjectiveSpace, Y, p, X) = copyto!(Y, X)

function exp!(M::AbstractProjectiveSpace, q, p, X)
    θ = norm(M, p, X)
    q .= cos(θ) .* p .+ usinc(θ) .* X
    return q
end

flat!(::AbstractProjectiveSpace, ξ::CoTFVector, p, X::TFVector) = copyto!(ξ, X)

function get_basis(::ProjectiveSpace{n,ℝ}, p, B::DiagonalizingOrthonormalBasis{ℝ}) where {n}
    return get_basis(Sphere{n,ℝ}(), p, B)
end

@doc raw"""
    get_coordinates(M::ProjectiveSpace, p, X, B::DefaultOrthonormalBasis)

Represent the tangent vector `X` at point `p` from the [`ProjectiveSpace`](@ref) `M` in
an orthonormal basis by rotating the vector `X` using the rotation matrix
$2\frac{q q^\mathrm{T}}{q^\mathrm{T} q} - I$ where $q = p + (1, 0, …, 0)$.
"""
function get_coordinates(::ProjectiveSpace{n,ℝ}, p, X, ::DefaultOrthonormalBasis) where {n}
    isapprox(p[1], 1) && return X[2:end]
    xp1 = p .+ ntuple(i -> ifelse(i == 1, 1, 0), n + 1)
    return ((2 * dot(xp1, X) / dot(xp1, xp1)) .* xp1 - X)[2:end]
end

function get_coordinates!(
    M::ProjectiveSpace{n,ℝ},
    Y,
    p,
    X,
    B::DefaultOrthonormalBasis,
) where {n}
    return copyto!(Y, get_coordinates(M, p, X, B))
end

function get_vector(::ProjectiveSpace{n,ℝ}, p, X, ::DefaultOrthonormalBasis) where {n}
    p[1] ≈ 1 && return vcat(0, X)
    xp1 = p .+ ntuple(i -> ifelse(i == 1, 1, 0), n + 1)
    X0 = vcat(0, X)
    return (2 * dot(xp1, X0) / dot(xp1, xp1)) .* xp1 .- X0
end

function get_vector!(
    M::ProjectiveSpace{n,ℝ},
    Y::AbstractVector,
    p,
    X,
    B::DefaultOrthonormalBasis,
) where {n}
    return copyto!(Y, get_vector(M, p, X, B))
end

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

function Base.isapprox(::AbstractProjectiveSpace, p, q; kwargs...)
    return isapprox(abs2(dot(p, q)), 1; kwargs...)
end

function log!(M::AbstractProjectiveSpace, X, p, q)
    z = dot(p, q)
    cosθ = abs(z)
    signz = sign_from_abs(z, cosθ)
    X .= (signz' .* q .- cosθ .* p) ./ usinc_from_cos(cosθ)
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

function mid_point!(M::ProjectiveSpace, q, p1, p2)
    z = dot(p1, p2)
    cosθ = abs(z)
    signz = sign_from_abs(z, cosθ)
    q .= p1 .+ signz' .* p2
    project!(M, q, q)
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

project!(::AbstractProjectiveSpace, q, p) = (q .= p ./ norm(p))

project!(::AbstractProjectiveSpace, Y, p, X) = (Y .= X .- dot(p, X) .* p)

@doc raw"""
    representation_size(M::AbstractProjectiveSpace)

Return the size points on the [`AbstractProjectiveSpace`](@ref) `M` are represented as,
i.e., the representation size of the embedding.
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

@doc raw"""
vector_transport_to(M::AbstractProjectiveSpace, p, X, q, method::ParallelTransport)

Parallel transport a vector `X` from the tangent space at a point `p` on the
[`AbstractProjectiveSpace`](@ref) `M` along the `shortest_geodesic`](@ref) to the tangent
space at another point `q`.
"""
vector_transport_to(::AbstractProjectiveSpace, p, X, q, ::ParallelTransport)

@doc raw"""
    vector_transport_to(M::AbstractProjectiveSpace, p, X, q, method::ProjectionTransport)

Transport a vector `X` from the tangent space at `p` on the
[`AbstractProjectiveSpace`](@ref) `M` by interpreting it as an element of the embedding and
then projecting it onto the tangent space at `q`.
"""
vector_transport_to(::AbstractProjectiveSpace, ::Any, ::Any, ::Any, ::ProjectionTransport)

@doc raw"""
    vector_transport_direction(M::AbstractProjectiveSpace, p, X, d, method::ParallelTransport)

Parallel transport a vector `X` from the tangent space at a point `p` on the
[`AbstractProjectiveSpace`](@ref) `M` along the geodesic in the direction indicated by the
tangent vector `d`.
"""
vector_transport_direction(::AbstractProjectiveSpace, p, X, d, ::ParallelTransport)

function vector_transport_to!(::AbstractProjectiveSpace, Y, p, X, q, ::ParallelTransport)
    z = dot(p, q)
    signz = sign(z)
    m = p .+ signz' .* q # un-normalized midpoint
    mnorm2 = real(dot(m, m))
    factor = signz * dot(q, X) * (2 / mnorm2)
    # multiply by `sign(z)` to bring from T_{\exp_p(\log_p q)} M to T_q M
    # this ensures that subsequent functions like `exp(M, q, Y)` do the right thing
    Y .= signz .* (X .- m .* factor)
    return Y
end
function vector_transport_to!(M::AbstractProjectiveSpace, Y, p, X, q, ::ProjectionTransport)
    project!(M, Y, q, X)
    return Y
end

function vector_transport_direction!(
    M::AbstractProjectiveSpace,
    Y,
    p,
    X,
    d,
    ::ParallelTransport,
)
    θ = norm(M, p, d)
    cosθ = cos(θ)
    dX = inner(M, p, d, X)
    α = usinc(θ) * dX
    β = ifelse(iszero(θ), zero(cosθ), (1 - cosθ) / θ^2) * dX
    Y .= X .- α .* p .- β .* d
    return Y
end
