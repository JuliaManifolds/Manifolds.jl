"""
    AbstractPowerRepresentation

An abstract representation type of points and tangent vectors on a power manifold.
"""
abstract type AbstractPowerRepresentation end

"""
    ArrayPowerRepresentation

Representation of points and tangent vectors on a power manifold using multidimensional
arrays where first dimensions are equal to [`representation_size`](@ref) of the
wrapped manifold and the following ones are equal to the number of elements in each
direction.

[`Torus`](@ref) uses this representation.
"""
struct ArrayPowerRepresentation <: AbstractPowerRepresentation end

"""
    NestedPowerRepresentation

Representation of points and tangent vectors on a power manifold using arrays
of size equal to `TSize` of a [`PowerManifold`](@ref).
Each element of such array stores a single point or tangent vector.

[`GraphManifold`](@ref) uses this representation.
"""
struct NestedPowerRepresentation <: AbstractPowerRepresentation end

@doc raw"""
    AbstractPowerManifold{𝔽,M,TPR} <: Manifold{𝔽}

An abstract [`Manifold`](@ref) to represent manifolds that are build as powers
of another [`Manifold`](@ref) `M` with representation type `TPR`, a subtype of
[`AbstractPowerRepresentation`](@ref).
"""
abstract type AbstractPowerManifold{𝔽,M<:Manifold{𝔽},TPR<:AbstractPowerRepresentation} <:
              Manifold{𝔽} end

@doc raw"""
    PowerManifold{𝔽,TM<:Manifold,TSize<:Tuple,TPR<:AbstractPowerRepresentation} <: AbstractPowerManifold{𝔽,TM}

The power manifold $\mathcal M^{n_1× n_2 × … × n_d}$ with power geometry
 `TSize` statically defines the number of elements along each axis.

For example, a manifold-valued time series would be represented by a power manifold with
$d$ equal to 1 and $n_1$ equal to the number of samples. A manifold-valued image
(for example in diffusion tensor imaging) would be represented by a two-axis power
manifold ($d=2$) with $n_1$ and $n_2$ equal to width and height of the image.

While the size of the manifold is static, points on the power manifold
would not be represented by statically-sized arrays. Operations on small
power manifolds might be faster if they are represented as [`ProductManifold`](@ref).

# Constructor

    PowerManifold(M, N_1, N_2, ..., N_d)
    PowerManifold(M, NestedPowerRepresentation(), N_1, N_2, ..., N_d)
    M^(N_1, N_2, ..., N_d)

Generate the power manifold $M^{N_1 × N_2 × … × N_d}$.
By default, the [`ArrayPowerRepresentation`](@ref) of points
and tangent vectors is used, although a different one, for example
[`NestedPowerRepresentation`](@ref), can be given as the second argument to the
constructor.
When `M` is a `PowerManifold` (not any [`AbstractPowerManifold`](@ref)) itself, given
dimensions will be appended to the dimensions already present, for example
`PowerManifold(PowerManifold(Sphere(2), 2), 3)` is equivalent to
`PowerManifold(Sphere(2), 2, 3)`. This feature preserves the representation of the inner
power manifold (unless it's explicitly overridden).
"""
struct PowerManifold{𝔽,TM<:Manifold{𝔽},TSize,TPR<:AbstractPowerRepresentation} <:
       AbstractPowerManifold{𝔽,TM,TPR}
    manifold::TM
end

function PowerManifold(M::Manifold{𝔽}, size::Integer...) where {𝔽}
    return PowerManifold{𝔽,typeof(M),Tuple{size...},ArrayPowerRepresentation}(M)
end
function PowerManifold(
    M::Manifold{𝔽},
    ::TPR,
    size::Integer...,
) where {𝔽,TPR<:AbstractPowerRepresentation}
    return PowerManifold{𝔽,typeof(M),Tuple{size...},TPR}(M)
end
function PowerManifold(
    M::PowerManifold{𝔽,TM,TSize,TPR},
    size::Integer...,
) where {𝔽,TM<:Manifold{𝔽},TSize,TPR<:AbstractPowerRepresentation}
    return PowerManifold{𝔽,TM,Tuple{TSize.parameters...,size...},TPR}(M.manifold)
end
function PowerManifold(
    M::PowerManifold{𝔽,TM,TSize},
    ::TPR,
    size::Integer...,
) where {𝔽,TM<:Manifold{𝔽},TSize,TPR<:AbstractPowerRepresentation}
    return PowerManifold{𝔽,TM,Tuple{TSize.parameters...,size...},TPR}(M.manifold)
end

@doc raw"""
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
struct InversePowerRetraction{TR<:AbstractInverseRetractionMethod} <:
       AbstractInverseRetractionMethod
    inverse_retraction::TR
end

"""
    PowerPointDistribution(M::AbstractPowerManifold, distribution)

Power distribution on manifold `M`, based on `distribution`.
"""
struct PowerPointDistribution{TM<:AbstractPowerManifold,TD<:MPointDistribution,TX} <:
       MPointDistribution{TM}
    manifold::TM
    distribution::TD
    point::TX
end

"""
    PowerFVectorDistribution([type::VectorBundleFibers], [x], distr)

Generates a random vector at a `point` from vector space (a fiber of a tangent
bundle) of type `type` using the power distribution of `distr`.

Vector space type and `point` can be automatically inferred from distribution `distr`.
"""
struct PowerFVectorDistribution{
    TSpace<:VectorBundleFibers{<:VectorSpaceType,<:AbstractPowerManifold},
    TD<:FVectorDistribution,
    TX,
} <: FVectorDistribution{TSpace,TX}
    type::TSpace
    point::TX
    distribution::TD
end

"""
    PowerVectorTransport(method::AbstractVectorTransportMethod)

Power vector transport method based on `method`. Works on [`AbstractPowerManifold`](@ref)s.
"""
struct PowerVectorTransport{TR<:AbstractVectorTransportMethod} <:
       AbstractVectorTransportMethod
    method::TR
end

"""
    PowerBasisData{TB<:AbstractArray}

Data storage for an array of basis data.
"""
struct PowerBasisData{TB<:AbstractArray}
    bases::TB
end

const PowerManifoldMultidimensional =
    AbstractPowerManifold{𝔽,<:Manifold{𝔽},ArrayPowerRepresentation} where {𝔽}
const PowerManifoldNested =
    AbstractPowerManifold{𝔽,<:Manifold{𝔽},NestedPowerRepresentation} where {𝔽}

_access_nested(x, i::Int) = x[i]
_access_nested(x, i::Tuple) = x[i...]

function allocate_result(M::PowerManifoldNested, f, x...)
    return [
        allocate_result(M.manifold, f, map(y -> _access_nested(y, i), x)...)
        for i in get_iterator(M)
    ]
end
function allocate_result(M::PowerManifoldNested, f::typeof(flat), w::TFVector, x)
    alloc = [allocate(_access_nested(w.data, i)) for i in get_iterator(M)]
    return FVector(CotangentSpace, alloc)
end
function allocate_result(M::PowerManifoldNested, f::typeof(get_vector), p, X)
    return [allocate_result(M.manifold, f, _access_nested(p, i)) for i in get_iterator(M)]
end
function allocate_result(M::PowerManifoldNested, f::typeof(sharp), w::CoTFVector, x)
    alloc = [allocate(_access_nested(w.data, i)) for i in get_iterator(M)]
    return FVector(TangentSpace, alloc)
end
function allocate_result(
    M::PowerManifoldNested,
    f::typeof(get_coordinates),
    p,
    X,
    B::AbstractBasis,
)
    return invoke(
        allocate_result,
        Tuple{Manifold,typeof(get_coordinates),Any,Any,typeof(B)},
        M,
        f,
        p,
        X,
        B,
    )
end

function allocation_promotion_function(M::AbstractPowerManifold, f, args::Tuple)
    return allocation_promotion_function(M.manifold, f, args)
end

Base.:^(M::Manifold, n) = PowerManifold(M, n...)

"""
    check_manifold_point(M::AbstractProductManifold, p; kwargs...)

Check whether `p` is a valid point on an [`AbstractPowerManifold`](@ref) `M`,
i.e. each element of `p` has to be a valid point on the base manifold.
If `p` is not a point on `M` a `CompositeManifoldError` consisting of all error messages of the
components, for which the tests fail is returned.

The tolerance for the last test can be set using the `kwargs...`.
"""
function check_manifold_point(M::AbstractPowerManifold, p; kwargs...)
    rep_size = representation_size(M.manifold)
    e = [
        (i, check_manifold_point(M.manifold, _read(M, rep_size, p, i); kwargs...))
        for i in get_iterator(M)
    ]
    errors = filter((x) -> !(x[2] === nothing), e)
    cerr = [ComponentManifoldError(er...) for er in errors]
    (length(errors) > 1) && return CompositeManifoldError(cerr)
    (length(errors) == 1) && return cerr[1]
    return nothing
end

"""
    check_tangent_vector(M::AbstractPowerManifold, p, X; check_base_point = true, kwargs... )

Check whether `X` is a tangent vector to `p` an the [`AbstractPowerManifold`](@ref)
`M`, i.e. atfer [`check_manifold_point`](@ref)`(M, p)`, and all projections to
base manifolds must be respective tangent vectors.
The optional parameter `check_base_point` indicates, whether to call [`check_manifold_point`](@ref)  for `p`.
If `X` is not a tangent vector to `p` on `M` a `CompositeManifoldError` consisting of all error
messages of the components, for which the tests fail is returned.


The tolerance for the last test can be set using the `kwargs...`.
"""
function check_tangent_vector(
    M::AbstractPowerManifold,
    p,
    X;
    check_base_point = true,
    kwargs...,
)
    if check_base_point
        mpe = check_manifold_point(M, p)
        mpe === nothing || return mpe
    end
    rep_size = representation_size(M.manifold)
    e = [
        (
            i,
            check_tangent_vector(
                M.manifold,
                _read(M, rep_size, p, i),
                _read(M, rep_size, X, i);
                kwargs...,
            ),
        ) for i in get_iterator(M)
    ]
    errors = filter((x) -> !(x[2] === nothing), e)
    cerr = [ComponentManifoldError(er...) for er in errors]
    (length(errors) > 1) && return CompositeManifoldError(cerr)
    (length(errors) == 1) && return cerr[1]
    return nothing
end

function det_local_metric(
    M::MetricManifold{PowerMetric,𝔽,<:AbstractPowerManifold{𝔽}},
    p::AbstractArray,
) where {𝔽}
    result = one(number_eltype(p))
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        result *= det_local_metric(M.manifold, _read(M, rep_size, p, i))
    end
    return result
end

@doc raw"""
    distance(M::AbstractPowerManifold, p, q)

Compute the distance between `q` and `p` on an [`AbstractPowerManifold`](@ref),
i.e. from the element wise distances the Forbenius norm is computed.
"""
function distance(M::AbstractPowerManifold, p, q)
    sum_squares = zero(number_eltype(p))
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        sum_squares +=
            distance(M.manifold, _read(M, rep_size, p, i), _read(M, rep_size, q, i))^2
    end
    return sqrt(sum_squares)
end

@doc raw"""
    exp(M::AbstractPowerManifold, p, X)

Compute the exponential map from `p` in direction `X` on the [`AbstractPowerManifold`](@ref) `M`,
which can be computed using the base manifolds exponential map elementwise.
"""
exp(::AbstractPowerManifold, ::Any...)

function exp!(M::AbstractPowerManifold, q, p, X)
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        exp!(
            M.manifold,
            _write(M, rep_size, q, i),
            _read(M, rep_size, p, i),
            _read(M, rep_size, X, i),
        )
    end
    return q
end

@doc raw"""
    flat(M::AbstractPowerManifold, p, X::FVector{TangentSpaceType})

use the musical isomorphism to transform the tangent vector `X` from the tangent space at
`p` on an [`AbstractPowerManifold`](@ref) `M` to a cotangent vector.
This can be done elementwise for each entry of `X` (and `p`).
"""
flat(::AbstractPowerManifold, ::Any...)

function flat!(M::AbstractPowerManifold, ξ::CoTFVector, p, X::TFVector)
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        flat!(
            M.manifold,
            FVector(CotangentSpace, _write(M, rep_size, ξ.data, i)),
            _read(M, rep_size, p, i),
            FVector(TangentSpace, _read(M, rep_size, X.data, i)),
        )
    end
    return ξ
end

function get_basis(M::AbstractPowerManifold, p, B::AbstractBasis)
    rep_size = representation_size(M.manifold)
    vs = [get_basis(M.manifold, _read(M, rep_size, p, i), B) for i in get_iterator(M)]
    return CachedBasis(B, PowerBasisData(vs))
end
function get_basis(M::AbstractPowerManifold, p, B::DiagonalizingOrthonormalBasis)
    rep_size = representation_size(M.manifold)
    vs = [
        get_basis(
            M.manifold,
            _read(M, rep_size, p, i),
            DiagonalizingOrthonormalBasis(_read(M, rep_size, B.frame_direction, i)),
        ) for i in get_iterator(M)
    ]
    return CachedBasis(B, PowerBasisData(vs))
end
for BT in ManifoldsBase.DISAMBIGUATION_BASIS_TYPES
    if BT == DiagonalizingOrthonormalBasis
        continue
    end
    eval(quote
        @invoke_maker 3 AbstractBasis get_basis(M::AbstractPowerManifold, p, B::$BT)
    end)
end

"""
    get_component(M::AbstractPowerManifold, p, idx...)

Get the component of a point `p` on an [`AbstractPowerManifold`](@ref) `M` at index `idx`.
"""
function get_component(M::AbstractPowerManifold, p, idx...)
    rep_size = representation_size(M.manifold)
    return _read(M, rep_size, p, idx)
end

function get_coordinates(M::AbstractPowerManifold, p, X, B::DefaultOrthonormalBasis)
    rep_size = representation_size(M.manifold)
    vs = [
        get_coordinates(M.manifold, _read(M, rep_size, p, i), _read(M, rep_size, X, i), B) for i in get_iterator(M)
    ]
    return reduce(vcat, reshape(vs, length(vs)))
end
function get_coordinates(
    M::AbstractPowerManifold,
    p,
    X,
    B::CachedBasis{𝔽,<:AbstractBasis,<:PowerBasisData},
) where {𝔽}
    rep_size = representation_size(M.manifold)
    vs = [
        get_coordinates(
            M.manifold,
            _read(M, rep_size, p, i),
            _read(M, rep_size, X, i),
            _access_nested(B.data.bases, i),
        ) for i in get_iterator(M)
    ]
    return reduce(vcat, reshape(vs, length(vs)))
end

function get_coordinates!(M::AbstractPowerManifold, Y, p, X, B::DefaultOrthonormalBasis)
    rep_size = representation_size(M.manifold)
    dim = manifold_dimension(M.manifold)
    v_iter = 1
    for i in get_iterator(M)
        # TODO: this view is really suboptimal when `dim` can be statically determined
        get_coordinates!(
            M.manifold,
            view(Y, v_iter:(v_iter + dim - 1)),
            _read(M, rep_size, p, i),
            _read(M, rep_size, X, i),
            B,
        )
        v_iter += dim
    end
    return Y
end
function get_coordinates!(
    M::AbstractPowerManifold,
    Y,
    p,
    X,
    B::CachedBasis{𝔽,<:AbstractBasis,<:PowerBasisData},
) where {𝔽}
    rep_size = representation_size(M.manifold)
    dim = manifold_dimension(M.manifold)
    v_iter = 1
    for i in get_iterator(M)
        get_coordinates!(
            M.manifold,
            view(Y, v_iter:(v_iter + dim - 1)),
            _read(M, rep_size, p, i),
            _read(M, rep_size, X, i),
            _access_nested(B.data.bases, i),
        )
        v_iter += dim
    end
    return Y
end

get_iterator(M::PowerManifold{𝔽,<:Manifold{𝔽},Tuple{N}}) where {𝔽,N} = Base.OneTo(N)
@generated function get_iterator(
    M::PowerManifold{𝔽,<:Manifold{𝔽},SizeTuple},
) where {𝔽,SizeTuple}
    size_tuple = size_to_tuple(SizeTuple)
    return Base.product(map(Base.OneTo, size_tuple)...)
end

function get_vector!(
    M::PowerManifold,
    Y,
    p,
    X,
    B::CachedBasis{𝔽,<:AbstractBasis{𝔽},<:PowerBasisData},
) where {𝔽}
    dim = manifold_dimension(M.manifold)
    rep_size = representation_size(M.manifold)
    v_iter = 1
    for i in get_iterator(M)
        get_vector!(
            M.manifold,
            _write(M, rep_size, Y, i),
            _read(M, rep_size, p, i),
            X[v_iter:(v_iter + dim - 1)],
            _access_nested(B.data.bases, i),
        )
        v_iter += dim
    end
    return Y
end
function get_vector!(M::AbstractPowerManifold, Y, p, X, B::DefaultOrthonormalBasis)
    dim = manifold_dimension(M.manifold)
    rep_size = representation_size(M.manifold)
    v_iter = 1
    for i in get_iterator(M)
        get_vector!(
            M.manifold,
            _write(M, rep_size, Y, i),
            _read(M, rep_size, p, i),
            X[v_iter:(v_iter + dim - 1)],
            B,
        )
        v_iter += dim
    end
    return Y
end

"""
    getindex(p, M::AbstractPowerManifold, i::Union{Integer,Colon,AbstractVector}...)
    p[M::AbstractPowerManifold, i...]

Access the element(s) at index `[i...]` of a point `p` on an [`AbstractPowerManifold`](@ref)
`M` by linear or multidimensional indexing.
See also [Array Indexing](https://docs.julialang.org/en/v1/manual/arrays/#man-array-indexing-1) in Julia.
"""
Base.@propagate_inbounds function Base.getindex(
    p::AbstractArray,
    M::AbstractPowerManifold,
    I::Union{Integer,Colon,AbstractVector}...,
)
    return get_component(M, p, I...)
end
Base.@propagate_inbounds function Base.getindex(
    p::AbstractArray,
    M::AbstractPowerManifold,
    I::Integer...,
)
    return collect(get_component(M, p, I...))
end

@doc raw"""
    injectivity_radius(M::AbstractPowerManifold[, p])

the injectivity radius on an [`AbstractPowerManifold`](@ref) is for the global case
equal to the one of its base manifold. For a given point `p` it's equal to the
minimum of all radii in the array entries.
"""
function injectivity_radius(M::AbstractPowerManifold, p)
    radius = 0.0
    initialized = false
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        cur_rad = injectivity_radius(M.manifold, _read(M, rep_size, p, i))
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
eval(
    quote
        @invoke_maker 1 Manifold injectivity_radius(
            M::AbstractPowerManifold,
            rm::AbstractRetractionMethod,
        )
    end,
)
eval(
    quote
        @invoke_maker 1 Manifold injectivity_radius(
            M::AbstractPowerManifold,
            rm::ExponentialRetraction,
        )
    end,
)

@doc raw"""
    inverse_retract(M::AbstractPowerManifold, p, q, m::InversePowerRetraction)

Compute the inverse retraction from `p` with respect to `q` on an [`AbstractPowerManifold`](@ref) `M`
using an [`InversePowerRetraction`](@ref), which by default encapsulates a inverse retraction
of the base manifold. Then this method is performed elementwise, so the encapsulated inverse
retraction method has to be one that is available on the base [`Manifold`](@ref).
"""
inverse_retract(::AbstractPowerManifold, ::Any...)

function inverse_retract!(M::AbstractPowerManifold, X, p, q, method::InversePowerRetraction)
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        inverse_retract!(
            M.manifold,
            _write(M, rep_size, X, i),
            _read(M, rep_size, p, i),
            _read(M, rep_size, q, i),
            method.inverse_retraction,
        )
    end
    return X
end

@doc raw"""
    inner(M::AbstractPowerManifold, p, X, Y)

Compute the inner product of `X` and `Y` from the tangent space at `p` on an
[`AbstractPowerManifold`](@ref) `M`, i.e. for each arrays entry the tangent
vector entries from `X` and `Y` are in the tangent space of the corresponding
element from `p`.
The inner product is then the sum of the elementwise inner products.
"""
function inner(M::AbstractPowerManifold, p, X, Y)
    result = zero(number_eltype(X))
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        result += inner(
            M.manifold,
            _read(M, rep_size, p, i),
            _read(M, rep_size, X, i),
            _read(M, rep_size, Y, i),
        )
    end
    return result
end

default_metric_dispatch(::AbstractPowerManifold, ::PowerMetric) = Val(true)

function Base.isapprox(M::AbstractPowerManifold, p, q; kwargs...)
    result = true
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        result &= isapprox(
            M.manifold,
            _read(M, rep_size, p, i),
            _read(M, rep_size, q, i);
            kwargs...,
        )
    end
    return result
end
function Base.isapprox(M::AbstractPowerManifold, p, X, Y; kwargs...)
    result = true
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        result &= isapprox(
            M.manifold,
            _read(M, rep_size, p, i),
            _read(M, rep_size, X, i),
            _read(M, rep_size, Y, i);
            kwargs...,
        )
    end
    return result
end

@doc raw"""
    log(M::AbstractPowerManifold, p, q)

Compute the logarithmic map from `p` to `q` on the [`AbstractPowerManifold`](@ref) `M`,
which can be computed using the base manifolds logarithmic map elementwise.
"""
log(::AbstractPowerManifold, ::Any...)

function log!(M::AbstractPowerManifold, X, p, q)
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        log!(
            M.manifold,
            _write(M, rep_size, X, i),
            _read(M, rep_size, p, i),
            _read(M, rep_size, q, i),
        )
    end
    return X
end


@doc raw"""
    manifold_dimension(M::PowerManifold)

Returns the manifold-dimension of an [`PowerManifold`](@ref) `M`
$=\mathcal N = (\mathcal M)^{n_1,…,n_d}$, i.e. with $n=(n_1,…,n_d)$ the array
size of the power manifold and $d_{\mathcal M}$ the dimension of the base manifold
$\mathcal M$, the manifold is of dimension

````math
\dim(\mathcal N) = \dim(\mathcal M)\prod_{i=1}^d n_i = n_1n_2\cdot…\cdot n_d \dim(\mathcal M).
````
"""
function manifold_dimension(M::PowerManifold{𝔽,<:Manifold,TSize}) where {𝔽,TSize}
    return manifold_dimension(M.manifold) * prod(size_to_tuple(TSize))
end

function mid_point!(M::AbstractPowerManifold, q, p1, p2)
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        mid_point!(
            M.manifold,
            _write(M, rep_size, q, i),
            _read(M, rep_size, p1, i),
            _read(M, rep_size, p2, i),
        )
    end
    return q
end

@doc raw"""
    norm(M::AbstractPowerManifold, p, X)

Compute the norm of `X` from the tangent space of `p` on an
[`AbstractPowerManifold`](@ref) `M`, i.e. from the element wise norms the
Frobenius norm is computed.
"""
function LinearAlgebra.norm(M::AbstractPowerManifold, p, X)
    sum_squares = zero(number_eltype(X))
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        sum_squares +=
            norm(M.manifold, _read(M, rep_size, p, i), _read(M, rep_size, X, i))^2
    end
    return sqrt(sum_squares)
end

@doc raw"""
    power_dimensions(M::PowerManifold)

return the power of `M`,
"""
function power_dimensions(M::PowerManifold{𝔽,<:Manifold,TSize}) where {𝔽,TSize}
    return size_to_tuple(TSize)
end

function Random.rand(rng::AbstractRNG, d::PowerFVectorDistribution)
    fv = zero_vector(d.type, d.point)
    Distributions._rand!(rng, d, fv)
    return fv
end
function Random.rand(rng::AbstractRNG, d::PowerPointDistribution)
    x = allocate_result(d.manifold, rand, d.point)
    Distributions._rand!(rng, d, x)
    return x
end

function Distributions._rand!(
    rng::AbstractRNG,
    d::PowerFVectorDistribution,
    v::AbstractArray,
)
    PM = d.type.manifold
    rep_size = representation_size(PM.manifold)
    for i in get_iterator(d.type.manifold)
        copyto!(d.distribution.point, _read(PM, rep_size, d.point, i))
        Distributions._rand!(rng, d.distribution, _read(PM, rep_size, v, i))
    end
    return v
end
function Distributions._rand!(rng::AbstractRNG, d::PowerPointDistribution, x::AbstractArray)
    M = d.manifold
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        Distributions._rand!(rng, d.distribution, _write(M, rep_size, x, i))
    end
    return x
end

Base.@propagate_inbounds @inline function _read(
    M::AbstractPowerManifold,
    rep_size::Tuple,
    x::AbstractArray,
    i::Int,
)
    return _read(M, rep_size, x, (i,))
end
Base.@propagate_inbounds @inline function _read(
    ::PowerManifoldMultidimensional,
    rep_size::Tuple,
    x::AbstractArray,
    i::Tuple,
)
    return view(x, rep_size_to_colons(rep_size)..., i...)
end
Base.@propagate_inbounds @inline function _read(
    ::PowerManifoldMultidimensional,
    rep_size::Tuple,
    x::HybridArray,
    i::Tuple,
)
    return x[rep_size_to_colons(rep_size)..., i...]
end
Base.@propagate_inbounds @inline function _read(
    ::PowerManifoldNested,
    rep_size::Tuple,
    x::AbstractArray,
    i::Tuple,
)
    return view(x[i...], rep_size_to_colons(rep_size)...)
end

@generated function rep_size_to_colons(rep_size::Tuple)
    N = length(rep_size.parameters)
    return ntuple(i -> Colon(), N)
end

function representation_size(M::PowerManifold{𝔽,<:Manifold,TSize}) where {𝔽,TSize}
    return (representation_size(M.manifold)..., size_to_tuple(TSize)...)
end

@doc raw"""
    retract(M::AbstractPowerManifold, p, X, method::PowerRetraction)

Compute the retraction from `p` with tangent vector `X` on an [`AbstractPowerManifold`](@ref) `M`
using a [`PowerRetraction`](@ref), which by default encapsulates a retraction of the
base manifold. Then this method is performed elementwise, so the encapsulated retraction
method has to be one that is available on the base [`Manifold`](@ref).
"""
retract(::AbstractPowerManifold, ::Any...)

function retract!(M::AbstractPowerManifold, q, p, X, method::PowerRetraction)
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        retract!(
            M.manifold,
            _write(M, rep_size, q, i),
            _read(M, rep_size, p, i),
            _read(M, rep_size, X, i),
            method.retraction,
        )
    end
    return q
end

"""
    set_component!(M::AbstractPowerManifold, q, p, idx...)

Set the component of a point `q` on an [`AbstractPowerManifold`](@ref) `M` at index `idx`
to `p`, which itself is a point on the [`Manifold`](@ref) the power manifold is build on.
"""
function set_component!(M::AbstractPowerManifold, q, p, idx...)
    rep_size = representation_size(M.manifold)
    return copyto!(_write(M, rep_size, q, idx), p)
end
"""
    setindex!(q, p, M::AbstractPowerManifold, i::Union{Integer,Colon,AbstractVector}...)
    q[M::AbstractPowerManifold, i...] = p

Set the element(s) at index `[i...]` of a point `q` on an [`AbstractPowerManifold`](@ref)
`M` by linear or multidimensional indexing to `q`.
See also [Array Indexing](https://docs.julialang.org/en/v1/manual/arrays/#man-array-indexing-1) in Julia.
"""
Base.@propagate_inbounds function Base.setindex!(
    q::AbstractArray,
    p,
    M::AbstractPowerManifold,
    I::Union{Integer,Colon,AbstractVector}...,
)
    return set_component!(M, q, p, I...)
end

@doc raw"""
    sharp(M::AbstractPowerManifold, p, ξ::FVector{CotangentSpaceType})

Use the musical isomorphism to transform the cotangent vector `ξ` from the tangent space at
`p` on an [`AbstractPowerManifold`](@ref) `M` to a tangent vector.
This can be done elementwise for every entry of `ξ` (and `p`).
"""
sharp(::AbstractPowerManifold, ::Any...)

function sharp!(M::AbstractPowerManifold, X::TFVector, p, ξ::CoTFVector)
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        sharp!(
            M.manifold,
            FVector(TangentSpace, _write(M, rep_size, X.data, i)),
            _read(M, rep_size, p, i),
            FVector(CotangentSpace, _read(M, rep_size, ξ.data, i)),
        )
    end
    return X
end

function Base.show(
    io::IO,
    mime::MIME"text/plain",
    B::CachedBasis{𝔽,T,D},
) where {T<:AbstractBasis,D<:PowerBasisData,𝔽}
    println(io, "$(T()) for a power manifold")
    for i in Base.product(map(Base.OneTo, size(B.data.bases))...)
        println(io, "Basis for component $i:")
        show(io, mime, _access_nested(B.data.bases, i))
        println(io)
    end
    return nothing
end
function Base.show(
    io::IO,
    M::PowerManifold{𝔽,TM,TSize,ArrayPowerRepresentation},
) where {𝔽,TM,TSize}
    return print(io, "PowerManifold($(M.manifold), $(join(TSize.parameters, ", ")))")
end
function Base.show(io::IO, M::PowerManifold{𝔽,TM,TSize,TPR}) where {𝔽,TM,TSize,TPR}
    return print(
        io,
        "PowerManifold($(M.manifold), $(TPR()), $(join(TSize.parameters, ", ")))",
    )
end

Distributions.support(tvd::PowerFVectorDistribution) = FVectorSupport(tvd.type, tvd.point)
Distributions.support(d::PowerPointDistribution) = MPointSupport(d.manifold)

function vector_bundle_transport(fiber::VectorSpaceType, M::PowerManifold)
    return PowerVectorTransport(ParallelTransport())
end

function vector_transport_direction(M::AbstractPowerManifold, p, X, d)
    return vector_transport_direction(M, p, X, d, PowerVectorTransport(ParallelTransport()))
end

function vector_transport_direction!(M::AbstractPowerManifold, Y, p, X, d)
    return vector_transport_direction!(
        M,
        Y,
        p,
        X,
        d,
        PowerVectorTransport(ParallelTransport()),
    )
end

@doc raw"""
    vector_transport_to(M::AbstractPowerManifold, p, X, q, method::PowerVectorTransport)

Compute the vector transport the tangent vector `X`at `p` to `q` on the
[`PowerManifold`](@ref) `M` using an [`PowerVectorTransport`](@ref) `m`.
This method is performed elementwise, i.e. the method `m` has to be implemented on the
base manifold.
"""
vector_transport_to(::AbstractPowerManifold, ::Any, ::Any, ::Any, ::PowerVectorTransport)
function vector_transport_to(M::AbstractPowerManifold, p, X, q)
    return vector_transport_to(M, p, X, q, PowerVectorTransport(ParallelTransport()))
end

function vector_transport_to!(M::AbstractPowerManifold, Y, p, X, q)
    return vector_transport_to!(M, Y, p, X, q, PowerVectorTransport(ParallelTransport()))
end
function vector_transport_to!(M::AbstractPowerManifold, Y, p, X, q, m::PowerVectorTransport)
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        vector_transport_to!(
            M.manifold,
            _write(M, rep_size, Y, i),
            _read(M, rep_size, p, i),
            _read(M, rep_size, X, i),
            _read(M, rep_size, q, i),
            m.method,
        )
    end
    return Y
end

"""
    view(p, M::AbstractPowerManifold, i::Union{Integer,Colon,AbstractVector}...)

Get the view of the element(s) at index `[i...]` of a point `p` on an
[`AbstractPowerManifold`](@ref) `M` by linear or multidimensional indexing.
"""
function Base.view(
    p::AbstractArray,
    M::AbstractPowerManifold,
    I::Union{Integer,Colon,AbstractVector}...,
)
    rep_size = representation_size(M.manifold)
    return _write(M, rep_size, p, I...)
end

@inline function _write(M::AbstractPowerManifold, rep_size::Tuple, x::AbstractArray, i::Int)
    return _write(M, rep_size, x, (i,))
end
@inline function _write(
    M::PowerManifoldMultidimensional,
    rep_size::Tuple,
    x::AbstractArray,
    i::Tuple,
)
    return view(x, rep_size_to_colons(rep_size)..., i...)
end
@inline function _write(M::PowerManifoldNested, rep_size::Tuple, x::AbstractArray, i::Tuple)
    return view(x[i...], rep_size_to_colons(rep_size)...)
end
