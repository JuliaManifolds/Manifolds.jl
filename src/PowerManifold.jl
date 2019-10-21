@doc doc"""
    PowerManifold{TM<:Manifold, TSize<:Tuple} <: Manifold

Power manifold $M^{N_1 \times N_2 \times \dots \times N_n}$ with power geometry.
It is represented by an array-like structure with $n$ dimensions and sizes
$N_1, N_2, \dots, N_n$, along each dimension.
`TSize` statically defines the number of elements along each axis.
For example, a manifold-valued time series would be represented by a power
manifold with $n$ equal to 1 and $N_1$ equal to the number of samples.
A manifold-valued image (for example in diffusion tensor imaging) would
be represented by a two-axis power manifold ($n=2$) with $N_1$ and $N_2$
equal to width and height of the image.

While the size of the manifold is static, points on the power manifold
would not be represented by statically-sized arrays. Operations on small
power manifolds might be faster if they are represented as [`ProductManifold`](@ref).

# Constructor

    PowerManifold(M, (N_1, N_2, ..., N_n))

generates the power manifold $M^{N_1 \times N_2 \times \dots \times N_n}$.
"""
struct PowerManifold{TM<:Manifold, TSize} <: Manifold
    manifold::TM
end

function PowerManifold(manifold::Manifold, size::Tuple)
    return PowerManifold{typeof(manifold), Tuple{size...}}(manifold)
end

function power_mapreduce(f, op, M::PowerManifold, kwargs::NamedTuple, init, x...)
    return power_mapreduce(f, op, M, kwargs, init, representation_size(M.manifold), x...)
end

function power_mapreduce(f, op, M::PowerManifold{<:Manifold, Tuple{N}}, kwargs::NamedTuple, init, rep_size::Tuple{Int}) where {N}
    return mapreduce(i -> f(M.manifold; kwargs...), op, Base.OneTo(N); init=init)
end

function power_mapreduce(f, op, M::PowerManifold{<:Manifold, Tuple{N}}, kwargs::NamedTuple, init, rep_size::Tuple{Int}, x1) where {N}
    return mapreduce(i -> f(M.manifold, view(x1, :, i); kwargs...), op, Base.OneTo(N); init=init)
end

function power_mapreduce(f, op, M::PowerManifold{<:Manifold, Tuple{N}}, kwargs::NamedTuple, init, rep_size::Tuple{Int}, x1, x2) where {N}
    return mapreduce(i -> f(M.manifold, view(x1, :, i), view(x2, :, i); kwargs...), op, Base.OneTo(N); init=init)
end

function power_mapreduce(f, op, M::PowerManifold{<:Manifold, Tuple{N}}, kwargs::NamedTuple, init, rep_size::Tuple{Int}, x1, x2, x3) where {N}
    return mapreduce(i -> f(M.manifold, view(x1, :, i), view(x2, :, i), view(x3, :, i); kwargs...), op, Base.OneTo(N); init=init)
end



function power_map(f, M::PowerManifold, kwargs::NamedTuple, x::AbstractArray...)
    return power_map(f, M, kwargs, representation_size(M.manifold), x...)
end

function power_map(f, M::PowerManifold{<:Manifold, Tuple{N}}, kwargs::NamedTuple, rep_size::Tuple{Int}) where {N}
    return map(i -> f(M.manifold; kwargs...), Base.OneTo(N))
end

function power_map(f, M::PowerManifold{<:Manifold, Tuple{N}}, kwargs::NamedTuple, rep_size::Tuple{Int}, x1::AbstractArray) where {N}
    return map(i -> f(M.manifold, view(x1, :, i); kwargs...), Base.OneTo(N))
end

function power_map(f, M::PowerManifold{<:Manifold, Tuple{N}}, kwargs::NamedTuple, rep_size::Tuple{Int}, x1::AbstractArray, x2::AbstractArray) where {N}
    return map(i -> f(M.manifold, view(x1, :, i), view(x2, :, i); kwargs...), Base.OneTo(N))
end

function power_map(f, M::PowerManifold{<:Manifold, Tuple{N}}, kwargs::NamedTuple, rep_size::Tuple{Int}, x1::AbstractArray, x2::AbstractArray, x3::AbstractArray) where {N}
    return map(i -> f(M.manifold, view(x1, :, i), view(x2, :, i), view(x3, :, i); kwargs...), Base.OneTo(N))
end


function isapprox(M::PowerManifold, x, y; kwargs...)
    return power_mapreduce(isapprox, &, M, kwargs.data, true, x, y)
end

function isapprox(M::PowerManifold, x, v, w; kwargs...)
    return power_mapreduce(isapprox, &, M, kwargs.data, true, x, v, w)
end

function representation_size(M::PowerManifold{<:Manifold, TSize}) where TSize
    return (representation_size(M.manifold)..., size_to_tuple(TSize)...)
end

function manifold_dimension(M::PowerManifold{<:Manifold, TSize}) where TSize
    return manifold_dimension(M.manifold)*prod(size_to_tuple(TSize))
end


struct PowerMetric <: Metric end

function det_local_metric(M::MetricManifold{PowerManifold, PowerMetric}, x::AbstractArray)
    dets = power_mapreduce(det_local_metric, +, M.manifold, NamedTuple(), 0, x)
    return prod(dets)
end

function inner(M::PowerManifold, x, v, w)
    return power_mapreduce(inner, +, M, NamedTuple(), 0, x, v, w)
end

function norm(M::PowerManifold, x, v)
    norms_squared = power_map(norm, M, NamedTuple(), x, v).^2
    return sqrt(sum(norms_squared))
end

function distance(M::PowerManifold, x, y)
    dists_squared = power_map(distance, M, NamedTuple(), x, y).^2
    return sqrt(sum(dists_squared))
end

function exp!(M::PowerManifold, y, x, v)
    power_map(exp!, M, NamedTuple(), y, x, v)
    return y
end

function log!(M::PowerManifold, v, x, y)
    power_map(log!, M, NamedTuple(), v, x, y)
    return v
end

function injectivity_radius(M::PowerManifold, x)
    radii = power_map(injectivity_radius, M, NamedTuple(), x)
    return minimum(radii)
end

function injectivity_radius(M::PowerManifold)
    radii = power_map(injectivity_radius, M, NamedTuple())
    return minimum(radii)
end

"""
    PowerRetraction(retraction::AbstractRetractionMethod)

Power retraction based on `retraction`. Works on [`PowerManifold`](@ref).
"""
struct PowerRetraction{TR<:AbstractRetractionMethod} <: AbstractRetractionMethod
    retraction::TR
end

function retract!(M::PowerManifold, y, x, v, method::PowerRetraction)
    power_map(M, NamedTuple(), y, x, v) do _M, _y, _x, _v
        retract!(_M, _y, _x, _v, method.retraction)
    end
    return y
end

"""
    InversePowerRetraction(inverse_retractions::AbstractInverseRetractionMethod...)

Power inverse retraction of `inverse_retractions`.
Works on [`PowerManifold`](@ref).
"""
struct InversePowerRetraction{TR<:AbstractInverseRetractionMethod} <: AbstractInverseRetractionMethod
    inverse_retraction::TR
end

function inverse_retract!(M::PowerManifold, v, x, y, method::InversePowerRetraction)
    power_map(M, NamedTuple(), v, x, y) do _M, _v, _x, _y
        inverse_retract!(_M, _v, _x, _y, method.inverse_retraction)
    end
    return v
end

function flat!(M::PowerManifold, v::FVector{CotangentSpaceType}, x, w::FVector{TangentSpaceType})
    error("TODO")
    return v
end

function sharp!(M::PowerManifold, v::FVector{TangentSpaceType}, x, w::FVector{CotangentSpaceType})
    error("TODO")
    return v
end

"""
    is_manifold_point(M::ProductManifold, x; kwargs...)

Check whether `x` is a valid point on the [`ProductManifold`](@ref) `M`.

The tolerance for the last test can be set using the ´kwargs...`.
"""
function is_manifold_point(M::PowerManifold, x; kwargs...)
    return power_mapreduce(is_manifold_point, &, M, kwargs.data, true, x)
end

"""
    is_tangent_vector(M::ProductManifold, x, v; kwargs... )

Check whether `v` is a tangent vector to `x` on the [`ProductManifold`](@ref)
`M`, i.e. atfer [`is_manifold_point`](@ref)`(M, x)`, and all projections to
base manifolds must be respective tangent vectors.

The tolerance for the last test can be set using the ´kwargs...`.
"""
function is_tangent_vector(M::PowerManifold, x, v; kwargs...)
    ispoint = is_manifold_point(M, x)
    return power_mapreduce(is_tangent_vector, &, M, kwargs.data, true, x, v)
end

"""
    PowerPointDistribution(M::PowerManifold, distribution)

Power distribution on manifold `M`, based on `distribution`.
"""
struct PowerPointDistribution{TM<:PowerManifold, TD<:MPointDistribution, TX} <: MPointDistribution{TM}
    manifold::TM
    distribution::TD
    x::TX
end

function support(d::PowerPointDistribution)
    return MPointSupport(d.manifold)
end

function rand(rng::AbstractRNG, d::PowerPointDistribution)
    x = similar(d.x)
    _rand!(rng, d, x)
    return x
end

function _rand!(rng::AbstractRNG, d::PowerPointDistribution, x::AbstractArray)
    power_map((M, u) -> _rand!(rng, d.distribution, u), d.manifold, NamedTuple(), x)
    return x
end

"""
    PowerFVectorDistribution([type::VectorBundleFibers], [x], distr)

Generates a random vector at point `x` from vector space (a fiber of a tangent
bundle) of type `type` using the power distribution of `distr`.

Vector space type and `x` can be automatically inferred from distribution `distr`.
"""
struct PowerFVectorDistribution{TSpace<:VectorBundleFibers{<:VectorSpaceType, <:PowerManifold}, TD<:(NTuple{N,Distribution} where N), TX} <: FVectorDistribution{TSpace, TX}
    type::TSpace
    x::TX
    distribution::TD
end

function PowerFVectorDistribution(type::VectorBundleFibers{<:VectorSpaceType, <:PowerManifold}, x::AbstractArray, distribution::FVectorDistribution)
    return PowerFVectorDistribution{typeof(type), typeof(distribution), typeof(x)}(type, x, distribution)
end

function PowerFVectorDistribution(type::VectorBundleFibers{<:VectorSpaceType, <:PowerManifold}, distribution::FVectorDistribution)
    error("TODO")
end

function PowerFVectorDistribution(distributions::FVectorDistribution)
    error("TODO")
end

function support(tvd::PowerFVectorDistribution)
    error("TODO")
    return FVectorSupport(tvd.type, ProductRepr(map(d -> support(d).x, tvd.distributions)...))
end

function rand(rng::AbstractRNG, d::PowerFVectorDistribution)
    error("TODO")
end

function _rand!(rng::AbstractRNG, d::PowerFVectorDistribution, v::AbstractArray)
    error("TODO")
    return v
end
