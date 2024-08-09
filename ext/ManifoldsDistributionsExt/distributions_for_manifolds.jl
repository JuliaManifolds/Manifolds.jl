
## product manifold

"""
    ProductPointDistribution(M::ProductManifold, distributions)

Product distribution on manifold `M`, combined from `distributions`.
"""
struct ProductPointDistribution{
    TM<:ProductManifold,
    TD<:(NTuple{N,Distribution} where {N}),
} <: MPointDistribution{TM}
    manifold::TM
    distributions::TD
end

function ProductPointDistribution(M::ProductManifold, distributions::MPointDistribution...)
    return ProductPointDistribution{typeof(M),typeof(distributions)}(M, distributions)
end
function ProductPointDistribution(distributions::MPointDistribution...)
    M = ProductManifold(map(d -> support(d).manifold, distributions)...)
    return ProductPointDistribution(M, distributions...)
end

"""
    ProductFVectorDistribution([type::VectorSpaceFiber], [x], distrs...)

Generates a random vector at point `x` from vector space (a fiber of a tangent
bundle) of type `type` using the product distribution of given distributions.

Vector space type and `x` can be automatically inferred from distributions `distrs`.
"""
struct ProductFVectorDistribution{
    TSpace<:VectorSpaceFiber{<:Any,<:ProductManifold},
    TD<:(NTuple{N,Distribution} where {N}),
} <: FVectorDistribution{TSpace}
    type::TSpace
    distributions::TD
end

function ProductFVectorDistribution(distributions::FVectorDistribution...)
    M = ProductManifold(map(d -> support(d).space.manifold, distributions)...)
    fiber_type = support(distributions[1]).space.fiber_type
    if !all(d -> support(d).space.fiber_type == fiber_type, distributions)
        error(
            "Not all distributions have support in vector spaces of the same type, which is currently not supported",
        )
    end
    # Probably worth considering sum spaces in the future?
    p = ArrayPartition(map(d -> support(d).space.point, distributions)...)
    return ProductFVectorDistribution(Fiber(M, p, fiber_type), distributions)
end

function Random.rand(rng::AbstractRNG, d::ProductFVectorDistribution)
    return ArrayPartition(map(d -> rand(rng, d), d.distributions)...)
end
function Random.rand(rng::AbstractRNG, d::ProductPointDistribution)
    return ArrayPartition(map(d -> rand(rng, d), d.distributions)...)
end

function Distributions._rand!(
    rng::AbstractRNG,
    d::ProductFVectorDistribution,
    X::ArrayPartition,
)
    map(
        (t1, t2) -> Distributions._rand!(rng, t1, t2),
        d.distributions,
        submanifold_components(d.type.manifold, X),
    )
    return X
end
function Distributions._rand!(
    rng::AbstractRNG,
    d::ProductPointDistribution,
    p::ArrayPartition,
)
    map(
        (t1, t2) -> Distributions._rand!(rng, t1, t2),
        d.distributions,
        submanifold_components(d.manifold, p),
    )
    return p
end

Distributions.support(d::ProductPointDistribution) = MPointSupport(d.manifold)
function Distributions.support(tvd::ProductFVectorDistribution)
    return FVectorSupport(tvd.type)
end

function uniform_distribution(M::ProductManifold)
    return ProductPointDistribution(M, map(uniform_distribution, M.manifolds))
end
function uniform_distribution(M::ProductManifold, p)
    return ProductPointDistribution(
        M,
        map(uniform_distribution, M.manifolds, submanifold_components(M, p)),
    )
end

## power manifold

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
    PowerFVectorDistribution([type::VectorSpaceFiber], [x], distr)

Generates a random vector at a `point` from vector space (a fiber of a tangent
bundle) of type `type` using the power distribution of `distr`.

Vector space type and `point` can be automatically inferred from distribution `distr`.
"""
struct PowerFVectorDistribution{TSpace<:VectorSpaceFiber,TD<:FVectorDistribution} <:
       FVectorDistribution{TSpace}
    type::TSpace
    distribution::TD
end

function Random.rand(rng::AbstractRNG, d::PowerFVectorDistribution)
    fv = zero_vector(d.type.manifold, d.type.point)
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
        copyto!(d.distribution.type.point, _read(PM, rep_size, d.type.point, i))
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

Distributions.support(tvd::PowerFVectorDistribution) = FVectorSupport(tvd.type)
Distributions.support(d::PowerPointDistribution) = MPointSupport(d.manifold)

# Rotations

"""
    NormalRotationDistribution(M::Rotations, d::Distribution, x::TResult)

Distribution that returns a random point on the manifold [`Rotations`](@ref)
`M`. Random point is generated using base distribution `d` and the type
of the result is adjusted to `TResult`.

See [`normal_rotation_distribution`](@ref) for details.
"""
struct NormalRotationDistribution{TResult,TM<:Rotations,TD<:Distribution} <:
       MPointDistribution{TM}
    manifold::TM
    distr::TD
end

function NormalRotationDistribution(
    M::Rotations,
    d::Distribution,
    x::TResult,
) where {TResult}
    return NormalRotationDistribution{TResult,typeof(M),typeof(d)}(M, d)
end

function normal_rotation_distribution(M::Rotations, p, σ::Real)
    n = get_parameter(M.size)[1]
    d = Distributions.MvNormal(zeros(n * n), σ * I)
    return NormalRotationDistribution(M, d, p)
end

function _fix_random_rotation(A::AbstractMatrix)
    s = diag(sign.(qr(A).R))
    D = Diagonal(s)
    C = qr(A).Q * D
    if det(C) < 0
        C[:, [1, 2]] = C[:, [2, 1]]
    end
    return C
end

function Random.rand(
    rng::AbstractRNG,
    d::NormalRotationDistribution{TResult,<:Rotations},
) where {TResult}
    n = get_parameter(d.manifold.size)[1]
    return if n == 1
        convert(TResult, ones(1, 1))
    else
        A = reshape(rand(rng, d.distr), (n, n))
        convert(TResult, _fix_random_rotation(A))
    end
end

function Distributions._rand!(
    rng::AbstractRNG,
    d::NormalRotationDistribution,
    x::AbstractArray{<:Real},
)
    return copyto!(x, rand(rng, d))
end

Distributions.support(d::NormalRotationDistribution) = MPointSupport(d.manifold)
