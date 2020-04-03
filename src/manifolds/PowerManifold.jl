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
    AbstractPowerManifold{M,TPR} <: Manifold

An abstract [`Manifold`](@ref) to represent manifolds that are build as powers
of another [`Manifold`](@ref) `M` with representation type `TPR`, a subtype of
[`AbstractPowerRepresentation`](@ref).
"""
abstract type AbstractPowerManifold{M<:Manifold,TPR<:AbstractPowerRepresentation} <:
              Manifold end

@doc raw"""
    PowerManifold{TM<:Manifold, TSize<:Tuple, TPR<:AbstractPowerRepresentation} <: AbstractPowerManifold{TM}

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

    PowerManifold(M, N_1, N_2, ..., N_n)
    PowerManifold(M, NestedPowerRepresentation(), N_1, N_2, ..., N_n)
    M^(N_1,N_2, ..., N_n)

Generate the power manifold $M^{N_1 × N_2 × … × N_n}$.
By default, the [`ArrayPowerRepresentation`](@ref) of points
and tangent vectors is used, although a different one, for example
[`NestedPowerRepresentation`](@ref), can be given as the second argument to the
constructor.
"""
struct PowerManifold{TM<:Manifold,TSize,TPR<:AbstractPowerRepresentation} <:
       AbstractPowerManifold{TM,TPR}
    manifold::TM
end

function PowerManifold(M::Manifold, size::Int...)
    return PowerManifold{typeof(M),Tuple{size...},ArrayPowerRepresentation}(M)
end
function PowerManifold(
    M::Manifold,
    ::TPR,
    size::Int...,
) where {TPR<:AbstractPowerRepresentation}
    PowerManifold{typeof(M),Tuple{size...},TPR}(M)
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
    PowerBasisData{TB<:AbstractArray}

Data storage for an array of basis data.
"""
struct PowerBasisData{TB<:AbstractArray}
    bases::TB
end

const POWER_BASIS_LIST_CACHED = [
    CachedBasis{<:AbstractBasis{ℝ},<:PowerBasisData},
    CachedBasis{<:ManifoldsBase.AbstractOrthogonalBasis{ℝ},<:PowerBasisData},
    CachedBasis{<:ManifoldsBase.AbstractOrthonormalBasis{ℝ},<:PowerBasisData},
    CachedBasis{<:AbstractBasis{ℂ},<:PowerBasisData},
]

const PowerManifoldMultidimensional =
    AbstractPowerManifold{<:Manifold,ArrayPowerRepresentation} where {TSize}
const PowerManifoldNested =
    AbstractPowerManifold{<:Manifold,NestedPowerRepresentation} where {TSize}

_access_nested(x, i::Int) = x[i]
_access_nested(x, i::Tuple) = x[i...]

function allocate_result(M::PowerManifoldNested, f, x...)
    return [allocate_result(M.manifold, f, map(y -> _access_nested(y, i), x)...) for i in get_iterator(M)]
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
function allocate_result(M::PowerManifoldNested, f::typeof(get_coordinates), p, X, B)
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

^(M::Manifold, n) = PowerManifold(M, n...)

"""
    check_manifold_point(M::AbstractProductManifold, p; kwargs...)

Check whether `p` is a valid point on an [`AbstractPowerManifold`](@ref) `M`,
i.e. each element of `p` has to be a valid point on the base manifold.

The tolerance for the last test can be set using the `kwargs...`.
"""
function check_manifold_point(M::AbstractPowerManifold, p; kwargs...)
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        imp = check_manifold_point(M.manifold, _read(M, rep_size, p, i); kwargs...)
        imp === nothing || return imp
    end
    return nothing
end

"""
    check_tangent_vector(M::AbstractPowerManifold, p, X; check_base_point = true, kwargs... )

Check whether `X` is a tangent vector to `p` an the [`AbstractPowerManifold`](@ref)
`M`, i.e. atfer [`check_manifold_point`](@ref)`(M, p)`, and all projections to
base manifolds must be respective tangent vectors.
The optional parameter `check_base_point` indicates, whether to call [`check_manifold_point`](@ref)  for `p`.
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
    for i in get_iterator(M)
        imp = check_tangent_vector(
            M.manifold,
            _read(M, rep_size, p, i),
            _read(M, rep_size, X, i);
            kwargs...,
        )
        imp === nothing || return imp
    end
    return nothing
end

function det_local_metric(
    M::MetricManifold{<:AbstractPowerManifold,PowerMetric},
    p::AbstractArray,
)
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

function get_coordinates(M::AbstractPowerManifold, p, X, B::DefaultOrthonormalBasis)
    rep_size = representation_size(M.manifold)
    vs = [
        get_coordinates(
            M.manifold,
            _read(M, rep_size, p, i),
            _read(M, rep_size, X, i),
            B,
        ) for i in get_iterator(M)
    ]
    return reduce(vcat, reshape(vs, length(vs)))
end
function get_coordinates(
    M::AbstractPowerManifold,
    p,
    X,
    B::CachedBasis{<:AbstractBasis,<:PowerBasisData,𝔽},
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
for BT in POWER_BASIS_LIST_CACHED
    eval(quote
        @invoke_maker 4 CachedBasis{<:AbstractBasis,<:PowerBasisData} get_coordinates(
            M::AbstractPowerManifold,
            p,
            X,
            B::$BT,
        )
    end)
end

function get_coordinates!(M::AbstractPowerManifold, Y, p, X, B::DefaultOrthonormalBasis)
    rep_size = representation_size(M.manifold)
    dim = manifold_dimension(M.manifold)
    v_iter = 1
    for i in get_iterator(M)
        # TODO: this view is really suboptimal when `dim` can be statically determined
        get_coordinates!(
            M.manifold,
            view(Y, v_iter:v_iter+dim-1),
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
    B::CachedBasis{<:AbstractBasis{ℝ},<:PowerBasisData,ℝ},
)
    TypeTuple = Tuple{
        AbstractPowerManifold,
        Any,
        Any,
        Any,
        CachedBasis{<:AbstractBasis,<:PowerBasisData,ℝ},
    }
    return invoke(get_coordinates!, TypeTuple, M, Y, p, X, B)
end
function get_coordinates!(
    M::AbstractPowerManifold,
    Y,
    p,
    X,
    B::CachedBasis{<:AbstractBasis,<:PowerBasisData,𝔽},
) where {𝔽}
    rep_size = representation_size(M.manifold)
    dim = manifold_dimension(M.manifold)
    v_iter = 1
    for i in get_iterator(M)
        get_coordinates!(
            M.manifold,
            view(Y, v_iter:v_iter+dim-1),
            _read(M, rep_size, p, i),
            _read(M, rep_size, X, i),
            _access_nested(B.data.bases, i),
        )
        v_iter += dim
    end
    return Y
end

get_iterator(M::PowerManifold{<:Manifold,Tuple{N}}) where {N} = 1:N
@generated function get_iterator(M::PowerManifold{<:Manifold,SizeTuple}) where {SizeTuple}
    size_tuple = size_to_tuple(SizeTuple)
    return Base.product(map(Base.OneTo, size_tuple)...)
end

function get_vector!(
    M::PowerManifold,
    Y,
    p,
    X,
    B::CachedBasis{<:AbstractBasis,<:PowerBasisData},
)
    dim = manifold_dimension(M.manifold)
    rep_size = representation_size(M.manifold)
    v_iter = 1
    for i in get_iterator(M)
        get_vector!(
            M.manifold,
            _write(M, rep_size, Y, i),
            _read(M, rep_size, p, i),
            X[v_iter:v_iter+dim-1],
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
            X[v_iter:v_iter+dim-1],
            B,
        )
        v_iter += dim
    end
    return Y
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
eval(quote
    @invoke_maker 1 Manifold injectivity_radius(
        M::AbstractPowerManifold,
        rm::AbstractRetractionMethod,
    )
end)
eval(quote
    @invoke_maker 1 Manifold injectivity_radius(
        M::AbstractPowerManifold,
        rm::ExponentialRetraction,
    )
end)

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

function isapprox(M::AbstractPowerManifold, p, q; kwargs...)
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
function isapprox(M::AbstractPowerManifold, p, X, Y; kwargs...)
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
function manifold_dimension(M::PowerManifold{<:Manifold,TSize}) where {TSize}
    return manifold_dimension(M.manifold) * prod(size_to_tuple(TSize))
end

@doc raw"""
    norm(M::AbstractPowerManifold, p, X)

Compute the norm of `X` from the tangent space of `p` on an
[`AbstractPowerManifold`](@ref) `M`, i.e. from the element wise norms the
Frobenius norm is computed.
"""
function norm(M::AbstractPowerManifold, p, X)
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
function power_dimensions(M::PowerManifold{<:Manifold,TSize}) where {TSize}
    return size_to_tuple(TSize)
end

@inline function _read(M::AbstractPowerManifold, rep_size::Tuple, x::AbstractArray, i::Int)
    return _read(M, rep_size, x, (i,))
end
@inline function _read(
    ::PowerManifoldMultidimensional,
    rep_size::Tuple,
    x::AbstractArray,
    i::Tuple,
)
    return view(x, rep_size_to_colons(rep_size)..., i...)
end
@inline function _read(
    ::PowerManifoldMultidimensional,
    rep_size::Tuple,
    x::HybridArray,
    i::Tuple,
)
    return x[rep_size_to_colons(rep_size)..., i...]
end
@inline function _read(::PowerManifoldNested, rep_size::Tuple, x::AbstractArray, i::Tuple)
    return view(x[i...], rep_size_to_colons(rep_size)...)
end

@generated function rep_size_to_colons(rep_size::Tuple)
    N = length(rep_size.parameters)
    return ntuple(i -> Colon(), N)
end

function representation_size(M::PowerManifold{<:Manifold,TSize}) where {TSize}
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

function show(
    io::IO,
    mime::MIME"text/plain",
    B::CachedBasis{T,D,𝔽},
) where {T<:AbstractBasis,D<:PowerBasisData,𝔽}
    println(io, "$(T()) for a power manifold")
    for i in Base.product(map(Base.OneTo, size(B.data.bases))...)
        println(io, "Basis for component $i:")
        show(io, mime, _access_nested(B.data.bases, i))
        println(io)
    end
end
function show(io::IO, M::PowerManifold{TM,TSize,ArrayPowerRepresentation}) where {TM,TSize}
    print(io, "PowerManifold($(M.manifold), $(join(TSize.parameters, ", ")))")
end
function show(io::IO, M::PowerManifold{TM,TSize,TPR}) where {TM,TSize,TPR}
    print(io, "PowerManifold($(M.manifold), $(TPR()), $(join(TSize.parameters, ", ")))")
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
