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

    PowerManifold(M, N_1, N_2, ..., N_n)

generates the power manifold $M^{N_1 \times N_2 \times \dots \times N_n}$.
"""
struct PowerManifold{TM<:Manifold, TSize} <: Manifold
    manifold::TM
end

@generated function rep_size_to_colons(rep_size::Tuple)
    N = length(rep_size.parameters)
    return ntuple(i -> Colon(), N)
end

function PowerManifold(manifold::Manifold, size::Int...)
    return PowerManifold{typeof(manifold), Tuple{size...}}(manifold)
end

function get_iterator(M::PowerManifold{<:Manifold, Tuple{N}}) where N
    return 1:N
end

@generated function get_iterator(M::PowerManifold{<:Manifold, SizeTuple}) where SizeTuple
    return Base.product(map(Base.OneTo, tuple(SizeTuple.parameters...))...)
end

@inline function _read(rep_size::Tuple, x::AbstractArray, i::Int)
    return _read(rep_size, x, (i,))
end

@inline function _read(rep_size::Tuple, x::AbstractArray, i::Tuple)
    return view(x, rep_size_to_colons(rep_size)..., i...)
end

@inline function _read(rep_size::Tuple, x::HybridArray, i::Tuple)
    return x[rep_size_to_colons(rep_size)..., i...]
end

@inline function _write(rep_size::Tuple, x::AbstractArray, i::Int)
    return _write(rep_size, x, (i,))
end

@inline function _write(rep_size::Tuple, x::AbstractArray, i::Tuple)
    return view(x, rep_size_to_colons(rep_size)..., i...)
end


function isapprox(M::PowerManifold, x, y; kwargs...)
    result = true
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        result &= isapprox(M.manifold,
            _read(rep_size, x, i),
            _read(rep_size, y, i);
            kwargs...)
    end
    return result
end

function isapprox(M::PowerManifold, x, v, w; kwargs...)
    result = true
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        result &= isapprox(M.manifold,
            _read(rep_size, x, i),
            _read(rep_size, v, i),
            _read(rep_size, w, i);
            kwargs...)
    end
    return result
end

function representation_size(M::PowerManifold{<:Manifold, TSize}) where TSize
    return (representation_size(M.manifold)..., size_to_tuple(TSize)...)
end

function manifold_dimension(M::PowerManifold{<:Manifold, TSize}) where TSize
    return manifold_dimension(M.manifold)*prod(size_to_tuple(TSize))
end


struct PowerMetric <: Metric end

function det_local_metric(M::MetricManifold{PowerManifold, PowerMetric}, x::AbstractArray)
    result = one(eltype(x))
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        result *= det_local_metric(M.manifold,
            _read(rep_size, x, i))
    end
    return result
end

function inner(M::PowerManifold, x, v, w)
    result = zero(eltype(v))
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        result += inner(M.manifold,
            _read(rep_size, x, i),
            _read(rep_size, v, i),
            _read(rep_size, w, i))
    end
    return result
end

function norm(M::PowerManifold, x, v)
    sum_squares = zero(eltype(v))
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        sum_squares += norm(M.manifold,
            _read(rep_size, x, i),
            _read(rep_size, v, i))^2
    end
    return sqrt(sum_squares)
end

function distance(M::PowerManifold, x, y)
    sum_squares = zero(eltype(x))
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        sum_squares += distance(M.manifold,
            _read(rep_size, x, i),
            _read(rep_size, y, i))^2
    end
    return sqrt(sum_squares)
end

function exp!(M::PowerManifold, y, x, v)
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        exp!(M.manifold,
            _write(rep_size, y, i),
            _read(rep_size, x, i),
            _read(rep_size, v, i))
    end
    return y
end

function log!(M::PowerManifold, v, x, y)
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        log!(M.manifold,
            _write(rep_size, v, i),
            _read(rep_size, x, i),
            _read(rep_size, y, i))
    end
    return v
end

function injectivity_radius(M::PowerManifold, x)
    radius = 0.0
    initialized = false
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        cur_rad = injectivity_radius(M.manifold,
            _read(rep_size, x, i))
        if initialized
            radius = min(cur_rad, radius)
        else
            radius = cur_rad
            initialized = true
        end
    end
    return radius
end

function injectivity_radius(M::PowerManifold)
    radius = 0.0
    initialized = false
    for i in get_iterator(M)
        cur_rad = injectivity_radius(M.manifold)
        if initialized
            radius = min(cur_rad, radius)
        else
            radius = cur_rad
            initialized = true
        end
    end
    return radius
end

"""
    PowerRetraction(retraction::AbstractRetractionMethod)

Power retraction based on `retraction`. Works on [`PowerManifold`](@ref).
"""
struct PowerRetraction{TR<:AbstractRetractionMethod} <: AbstractRetractionMethod
    retraction::TR
end

function retract!(M::PowerManifold, y, x, v, method::PowerRetraction)
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        retract!(M.manifold,
            _write(rep_size, y, i),
            _read(rep_size, x, i),
            _read(rep_size, v, i),
            method.retraction)
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
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        inverse_retract!(M.manifold,
            _write(rep_size, v, i),
            _read(rep_size, x, i),
            _read(rep_size, y, i),
            method.inverse_retraction)
    end
    return v
end

function flat!(M::PowerManifold, v::FVector{CotangentSpaceType}, x, w::FVector{TangentSpaceType})
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        flat!(M.manifold,
            FVector(CotangentSpace, _write(rep_size, v.data, i)),
            _read(rep_size, x, i),
            FVector(TangentSpace, _read(rep_size, w.data, i)))
    end
    return v
end

function sharp!(M::PowerManifold, v::FVector{TangentSpaceType}, x, w::FVector{CotangentSpaceType})
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        sharp!(M.manifold,
            FVector(TangentSpace, _write(rep_size, v.data, i)),
            _read(rep_size, x, i),
            FVector(CotangentSpace, _read(rep_size, w.data, i)))
    end
    return v
end

"""
    manifold_point_error(M::ProductManifold, x; kwargs...)

Check whether `x` is a valid point on the [`ProductManifold`](@ref) `M`.

The tolerance for the last test can be set using the ´kwargs...`.
"""
function manifold_point_error(M::PowerManifold, x; kwargs...)
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        imp = manifold_point_error(M.manifold,
            _read(rep_size, x, i); kwargs...)
        imp === nothing || return imp
    end
    return nothing
end

"""
    tangent_vector_error(M::ProductManifold, x, v; kwargs... )

Check whether `v` is a tangent vector to `x` on the [`ProductManifold`](@ref)
`M`, i.e. atfer [`is_manifold_point`](@ref)`(M, x)`, and all projections to
base manifolds must be respective tangent vectors.

The tolerance for the last test can be set using the ´kwargs...`.
"""
function tangent_vector_error(M::PowerManifold, x, v; kwargs...)
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        imp = tangent_vector_error(M.manifold,
            _read(rep_size, x, i),
            _read(rep_size, v, i); kwargs...)
        imp === nothing || return imp
    end
    return nothing
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
    rep_size = representation_size(d.manifold.manifold)
    for i in get_iterator(d.manifold)
        _rand!(rng,
            d.distribution,
            _write(rep_size, x, i))
    end
    return x
end

"""
    PowerFVectorDistribution([type::VectorBundleFibers], [x], distr)

Generates a random vector at point `x` from vector space (a fiber of a tangent
bundle) of type `type` using the power distribution of `distr`.

Vector space type and `x` can be automatically inferred from distribution `distr`.
"""
struct PowerFVectorDistribution{TSpace<:VectorBundleFibers{<:VectorSpaceType, <:PowerManifold}, TD<:FVectorDistribution, TX} <: FVectorDistribution{TSpace, TX}
    type::TSpace
    x::TX
    distribution::TD
end

function support(tvd::PowerFVectorDistribution)
    return FVectorSupport(tvd.type, tvd.x)
end

function rand(rng::AbstractRNG, d::PowerFVectorDistribution)
    fv = zero_vector(d.type, d.x)
    _rand!(rng, d, fv)
    return fv
end

function _rand!(rng::AbstractRNG, d::PowerFVectorDistribution, v::AbstractArray)
    rep_size = representation_size(d.type.M.manifold)
    for i in get_iterator(d.type.M)
        copyto!(d.distribution.x, _read(rep_size, d.x, i))
        _rand!(rng, d.distribution, _read(rep_size, v, i))
    end
    return v
end
