@doc doc"""
    AbstractPowerManifold{M,TSize} <: Manifold

An abstract [`Manifold`](@ref) to represent manifolds that are build as powers
of another [`Manifold`](@ref) `M` to the power `TSize`.
"""
abstract type AbstractPowerManifold{M <: Manifold} <: Manifold end

@doc doc"""
    PowerManifold{TM<:Manifold, TSize<:Tuple} <: AbstractPowerManifold

The power manifold $\mathcal M^{n_1 \times n_2 \times \dots \times n_d}$ with power geometry
represented by an array-like structure with $d$ dimensions and sizes $n_1, n_2, \dots, n_d$,
along each dimension. `TSize` statically defines the number of elements along each axis.
For example, a manifold-valued time series would be represented by a power manifold with
$d$ equal to 1 and $n_1$ equal to the number of samples. A manifold-valued image
(for example in diffusion tensor imaging) would be represented by a two-axis power
manifold ($d=2$) with $n_1$ and $n_2$ equal to width and height of the image.

While the size of the manifold is static, points on the power manifold
would not be represented by statically-sized arrays. Operations on small
power manifolds might be faster if they are represented as [`ProductManifold`](@ref).

# Constructor

    PowerManifold(M, N_1, N_2, ..., N_n)

Generate the power manifold $M^{N_1 \times N_2 \times \dots \times N_n}$.
"""
struct PowerManifold{TM, TSize} <: AbstractPowerManifold{TM}
    manifold::TM
end
PowerManifold(M::Manifold, size::Int...) = PowerManifold{typeof(M), Tuple{size...}}(M)

@doc doc"""
    PowerMetric <: Metric

Represent the [`Metric`](@ref) on an [`AbstractPowerManifold`](@ref), i.e. the inner
product on the tangent space is the sum of the inner product of each elements
tangent space of the power manifold.
"""
struct PowerMetric <: Metric end

"""
    PowerRetraction(retraction::AbstractRetractionMethod)

Power retraction based on `retraction`. Works on [`AbstractPowerManifold`](@ref)s.
"""
struct PowerRetraction{TR<:AbstractRetractionMethod} <: AbstractRetractionMethod
    retraction::TR
end
"""
    InversePowerRetraction(inverse_retractions::AbstractInverseRetractionMethod...)

Power inverse retraction of `inverse_retractions`. Works on [`AbstractPowerManifold`](@ref)s.
"""
struct InversePowerRetraction{TR<:AbstractInverseRetractionMethod} <: AbstractInverseRetractionMethod
    inverse_retraction::TR
end

"""
    PowerPointDistribution(M::AbstractPowerManifold, distribution)

Power distribution on manifold `M`, based on `distribution`.
"""
struct PowerPointDistribution{TM<:AbstractPowerManifold, TD<:MPointDistribution, TX} <: MPointDistribution{TM}
    manifold::TM
    distribution::TD
    x::TX
end

"""
    PowerFVectorDistribution([type::VectorBundleFibers], [x], distr)

Generates a random vector at point `x` from vector space (a fiber of a tangent
bundle) of type `type` using the power distribution of `distr`.

Vector space type and `x` can be automatically inferred from distribution `distr`.
"""
struct PowerFVectorDistribution{
        TSpace<:VectorBundleFibers{<:VectorSpaceType, <:AbstractPowerManifold},
        TD<:FVectorDistribution, TX
    } <: FVectorDistribution{TSpace, TX}
    type::TSpace
    x::TX
    distribution::TD
end

"""
    check_manifold_point(M::AbstractProductManifold, x; kwargs...)

Check whether `x` is a valid point on an [`AbstractPowerManifold`](@ref) `M`, i.e.
each element of `x` has to be a valid point on the base manifold.

The tolerance for the last test can be set using the `kwargs...`.
"""
function check_manifold_point(M::AbstractPowerManifold, x; kwargs...)
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        imp = check_manifold_point(M.manifold,
            _read(rep_size, x, i); kwargs...)
        imp === nothing || return imp
    end
    return nothing
end

"""
    check_tangent_vector(M::AbstractPowerManifold, x, v; kwargs... )

Check whether `v` is a tangent vector to `x` an the [`AbstractPowerManifold`](@ref)
`M`, i.e. atfer [`check_manifold_point`](@ref)`(M, x)`, and all projections to
base manifolds must be respective tangent vectors.

The tolerance for the last test can be set using the `kwargs...`.
"""
function check_tangent_vector(M::AbstractPowerManifold, x, v; kwargs...)
    mpe = check_manifold_point(M, x)
    if mpe !== nothing
        return mpe
    end
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        imp = check_tangent_vector(M.manifold,
            _read(rep_size, x, i),
            _read(rep_size, v, i); kwargs...)
        imp === nothing || return imp
    end
    return nothing
end

function det_local_metric(M::MetricManifold{<:AbstractPowerManifold, PowerMetric}, x::AbstractArray)
    result = one(eltype(x))
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        result *= det_local_metric(M.manifold,
            _read(rep_size, x, i))
    end
    return result
end

@doc doc"""
    distance(M::AbstractPowerManifold, x, y)

Compute the distance between `x` and `y` on an [`AbstractPowerManifold`](@ref),
i.e. from the element wise distances the Forbenius norm is computed.
"""
function distance(M::AbstractPowerManifold, x, y)
    sum_squares = zero(eltype(x))
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        sum_squares += distance(M.manifold,
            _read(rep_size, x, i),
            _read(rep_size, y, i))^2
    end
    return sqrt(sum_squares)
end

@doc doc"""
    exp(M::AbstractPowerManifold, x, v)

Compute the exponential map from `x` in direction `v` on the [`AbstractPowerManifold`](@ref) `M`,
which can be computed using the base manifolds exponential map elementwise.
"""
exp(::AbstractPowerManifold, ::Any...)
function exp!(M::AbstractPowerManifold, y, x, v)
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        exp!(M.manifold,
            _write(rep_size, y, i),
            _read(rep_size, x, i),
            _read(rep_size, v, i))
    end
    return y
end

@doc doc"""
    flat(M::AbstractPowerManifold, x, w::FVector{TangentSpaceType})

use the musical isomorphism to transform the tangent vector `w` from the tangent space at
`x` on an [`AbstractPowerManifold`](@ref) `M` to a cotangent vector.
This can be done elementwise, so r every entry of `w` (and `x`) sparately
"""
flat(::AbstractPowerManifold, ::Any...)
function flat!(M::AbstractPowerManifold, v::FVector{CotangentSpaceType}, x, w::FVector{TangentSpaceType})
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        flat!(M.manifold,
            FVector(CotangentSpace, _write(rep_size, v.data, i)),
            _read(rep_size, x, i),
            FVector(TangentSpace, _read(rep_size, w.data, i)))
    end
    return v
end

function get_iterator(M::PowerManifold{<:Manifold, Tuple{N}}) where N
    return 1:N
end
@generated function get_iterator(M::PowerManifold{<:Manifold, SizeTuple}) where SizeTuple
    return Base.product(map(Base.OneTo, tuple(SizeTuple.parameters...))...)
end

@doc doc"""
    injectivity_radius(M::AbstractPowerManifold[, x])

the injectivity radius on an [`AbstractPowerManifold`](@ref) is for the global case
equal to the one of its base manifold. For a given point `x` it's equal to the
minimum of all radii in the array entries.
"""
function injectivity_radius(M::AbstractPowerManifold, x)
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
injectivity_radius(M::AbstractPowerManifold) = injectivity_radius(M.manifold)

@doc doc"""
    inverse_retract(M::AbstractPowerManifold, x, y, m::InversePowerRetraction)

Compute the inverse retraction from `x` with respect to `y` on an [`AbstractPowerManifold`](@ref) `M`
using an [`InversePowerRetraction`](@ref), which by default encapsulates a inverse retraction
of the base manifold. Then this method is performed elementwise, so the encapsulated inverse
retraction method has to be one that is available on the base [`Manifold`](@ref).
"""
inverse_retract(::AbstractPowerManifold, ::Any...)
function inverse_retract!(M::AbstractPowerManifold, v, x, y, method::InversePowerRetraction)
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

@doc doc"""
    inner(M::AbstractPowerManifold, x, v, w)

Compute the inner product of `v` and `w` from the tangent space at `x` on an
[`AbstractPowerManifold`](@ref) `M`, i.e. for each arrays entry the tangent
vector entries from `v` and `w` are in the tangent space of the corresponding
element from `x`.
The inner product is then the sum of the elementwise inner products.
"""
function inner(M::AbstractPowerManifold, x, v, w)
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

is_default_metric(::AbstractPowerManifold,::PowerMetric) = Val(true)

function isapprox(M::AbstractPowerManifold, x, y; kwargs...)
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
function isapprox(M::AbstractPowerManifold, x, v, w; kwargs...)
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

@doc doc"""
    log(M::AbstractPowerManifold, x, y)

Compute the logarithmic map from `x` to `y` on the [`AbstractPowerManifold`](@ref) `M`,
which can be computed using the base manifolds logarithmic map elementwise.
"""
log(::AbstractPowerManifold, ::Any...)
function log!(M::AbstractPowerManifold, v, x, y)
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        log!(M.manifold,
            _write(rep_size, v, i),
            _read(rep_size, x, i),
            _read(rep_size, y, i))
    end
    return v
end


@doc doc"""
    manifold_dimension(M::PowerManifold)

Returns the manifold-dimension of an [`PowerManifold`](@ref) `M`
$=\mathcal N = (\mathcal M)^{n_1,\ldots,n_d}, i.e. with $n=(n_1,\ldots,n_d)$ the array
size of the power manifold and $d_{\mathcal M}$ the dimension of the base manifold
$\mathcal M$, the manifold is of dimension

````math
d_{\mathcal N} = d_{\mathcal M}\prod_{i=1}^d n_i = n_1n_2\cdot\ldots\cdotn_dd_{\mathcal M}.
````
"""
function manifold_dimension(M::PowerManifold{<:Manifold, TSize}) where {TSize}
    return manifold_dimension(M.manifold)*prod(size_to_tuple(TSize))
end

@doc doc"""
    norm(M::AbstractPowerManifold, x, v)

Compute the norm of `v` from the tangent space of `x` on an
[`AbstractPowerManifold`](@ref) `M`, i.e. from the element wise norms the
Frobenius norm is computed.
"""
function norm(M::AbstractPowerManifold, x, v)
    sum_squares = zero(eltype(v))
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        sum_squares += norm(M.manifold,
            _read(rep_size, x, i),
            _read(rep_size, v, i))^2
    end
    return sqrt(sum_squares)
end

function rand(rng::AbstractRNG, d::PowerFVectorDistribution)
    fv = zero_vector(d.type, d.x)
    _rand!(rng, d, fv)
    return fv
end
function rand(rng::AbstractRNG, d::PowerPointDistribution)
    x = similar(d.x)
    _rand!(rng, d, x)
    return x
end
function _rand!(rng::AbstractRNG, d::PowerFVectorDistribution, v::AbstractArray)
    rep_size = representation_size(d.type.M.manifold)
    for i in get_iterator(d.type.M)
        copyto!(d.distribution.x, _read(rep_size, d.x, i))
        _rand!(rng, d.distribution, _read(rep_size, v, i))
    end
    return v
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

@inline function _read(rep_size::Tuple, x::AbstractArray, i::Int)
    return _read(rep_size, x, (i,))
end
@inline function _read(rep_size::Tuple, x::AbstractArray, i::Tuple)
    return view(x, rep_size_to_colons(rep_size)..., i...)
end
@inline function _read(rep_size::Tuple, x::HybridArray, i::Tuple)
    return x[rep_size_to_colons(rep_size)..., i...]
end

@generated function rep_size_to_colons(rep_size::Tuple)
    N = length(rep_size.parameters)
    return ntuple(i -> Colon(), N)
end

function representation_size(M::PowerManifold{<:Manifold, TSize}) where {TSize}
    return (representation_size(M.manifold)..., size_to_tuple(TSize)...)
end

@doc doc"""
    retract(M::AbstractPowerManifold, x, v, m::PowerRetraction)

Compute the retraction from `x` with tangent vector `v` on an [`AbstractPowerManifold`](@ref) `M`
using an [`PowerRetraction`](@ref), which by default encapsulates a retraction of the
base manifold. Then this method is performed elementwise, so the encapsulated retraction
method has to be one that is available on the base [`Manifold`](@ref).
"""
retract(::AbstractPowerManifold, ::Any...)
function retract!(M::AbstractPowerManifold, y, x, v, method::PowerRetraction)
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

@doc doc"""
    sharp(M::AbstractPowerManifold, x, w::FVector{CotangentSpaceType})

Use the musical isomorphism to transform the cotangent vector `w` from the tangent space at
`x` on an [`AbstractPowerManifold`](@ref) `M` to a tangent vector.
This can be done elementwise, so for every entry of `w` (and `x`) sparately
"""
sharp(::AbstractPowerManifold, ::Any...)
function sharp!(M::AbstractPowerManifold, v::FVector{TangentSpaceType}, x, w::FVector{CotangentSpaceType})
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        sharp!(M.manifold,
            FVector(TangentSpace, _write(rep_size, v.data, i)),
            _read(rep_size, x, i),
            FVector(CotangentSpace, _read(rep_size, w.data, i)))
    end
    return v
end

support(tvd::PowerFVectorDistribution) = FVectorSupport(tvd.type, tvd.x)
support(d::PowerPointDistribution) = MPointSupport(d.manifold)

@inline function _write(rep_size::Tuple, x::AbstractArray, i::Int)
    return _write(rep_size, x, (i,))
end
@inline function _write(rep_size::Tuple, x::AbstractArray, i::Tuple)
    return view(x, rep_size_to_colons(rep_size)..., i...)
end
