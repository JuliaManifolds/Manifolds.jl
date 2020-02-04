@doc raw"""
    ProductManifold{TM<:Tuple, TRanges<:Tuple, TSizes<:Tuple} <: Manifold

Product manifold $M_1 × M_2 × …  × M_n$ with product geometry.
`TRanges` and `TSizes` statically define the relationship between representation
of the product manifold and representations of point, tangent vectors
and cotangent vectors of respective manifolds.

# Constructor

    ProductManifold(M_1, M_2, ..., M_n)

generates the product manifold $M_1 × M_2 × … × M_n$.
Alternatively, the same manifold can be contructed using the `×` operator:
`M_1 × M_2 × M_3`.
"""
struct ProductManifold{TM<:Tuple} <: Manifold
    manifolds::TM
end

ProductManifold(manifolds::Manifold...) = ProductManifold{typeof(manifolds)}(manifolds)
ProductManifold() = throw(MethodError("No method matching ProductManifold()."))

struct ProductMetric <: Metric end

"""
    ProductFVectorDistribution([type::VectorBundleFibers], [x], distrs...)

Generates a random vector at point `x` from vector space (a fiber of a tangent
bundle) of type `type` using the product distribution of given distributions.

Vector space type and `x` can be automatically inferred from distributions `distrs`.
"""
struct ProductFVectorDistribution{
    TSpace<:VectorBundleFibers{<:VectorSpaceType,<:ProductManifold},
    TD<:(NTuple{N,Distribution} where {N}),
    TX,
} <: FVectorDistribution{TSpace,TX}
    type::TSpace
    x::TX
    distributions::TD
end

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

"""
    ProductRetraction(retractions::AbstractRetractionMethod...)

Product retraction of `retractions`. Works on [`ProductManifold`](@ref).
"""
struct ProductRetraction{TR<:Tuple} <: AbstractRetractionMethod
    retractions::TR
end

function ProductRetraction(retractions::AbstractRetractionMethod...)
    return ProductRetraction{typeof(retractions)}(retractions)
end

"""
    InverseProductRetraction(retractions::AbstractInverseRetractionMethod...)

Product inverse retraction of `inverse retractions`. Works on [`ProductManifold`](@ref).
"""
struct InverseProductRetraction{TR<:Tuple} <: AbstractInverseRetractionMethod
    inverse_retractions::TR
end

function InverseProductRetraction(inverse_retractions::AbstractInverseRetractionMethod...)
    return InverseProductRetraction{typeof(inverse_retractions)}(inverse_retractions)
end

"""
    PrecomputedProductOrthonormalBasis(parts::NTuple{N,AbstractPrecomputedOrthonormalBasis} where N, F::AbstractNumbers = ℝ)

A precomputed orthonormal basis of a tangent space of a product manifold.
The tuple `parts` stores bases corresponding to multiplied manifolds.

The type parameter `F` denotes the [`AbstractNumbers`](@ref) that will be used as scalars.
"""
struct PrecomputedProductOrthonormalBasis{
    T<:NTuple{N,AbstractPrecomputedOrthonormalBasis} where {N},
    F,
} <: AbstractPrecomputedOrthonormalBasis{F}
    parts::T
end

function PrecomputedProductOrthonormalBasis(
    parts::NTuple{N,AbstractPrecomputedOrthonormalBasis},
    F::AbstractNumbers = ℝ,
) where {N}
    return PrecomputedProductOrthonormalBasis{typeof(parts),F}(parts)
end

"""
    check_manifold_point(M::ProductManifold, p; kwargs...)

Check whether `p` is a valid point on the [`ProductManifold`](@ref) `M`.

The tolerance for the last test can be set using the `kwargs...`.
"""
function check_manifold_point(M::ProductManifold, p::ProductRepr; kwargs...)
    for t ∈ ziptuples(M.manifolds, submanifold_components(M, p))
        err = check_manifold_point(t...; kwargs...)
        err === nothing || return err
    end
    return nothing
end

function check_manifold_point(M::ProductManifold, p::ProductArray; kwargs...)
    for t ∈ ziptuples(M.manifolds, submanifold_components(M, p))
        err = check_manifold_point(t...; kwargs...)
        err === nothing || return err
    end
    return nothing
end

"""
    check_tangent_vector(M::ProductManifold, p, X; kwargs... )

Check whether `X` is a tangent vector to `p` on the [`ProductManifold`](@ref)
`M`, i.e. after [`check_manifold_point`](@ref)`(M, p)`, and all projections to
base manifolds must be respective tangent vectors.

The tolerance for the last test can be set using the `kwargs...`.
"""
function check_tangent_vector(M::ProductManifold, p::ProductRepr, X::ProductRepr; kwargs...)
    perr = check_manifold_point(M, p)
    perr === nothing || return perr
    ts = ziptuples(M.manifolds, submanifold_components(M, p), submanifold_components(M, X))
    for t ∈ ts
        err = check_tangent_vector(t...; kwargs...)
        err === nothing || return err
    end
    return nothing
end
function check_tangent_vector(
    M::ProductManifold,
    p::ProductArray,
    X::ProductArray;
    kwargs...,
)
    perr = check_manifold_point(M, p)
    perr === nothing || return perr
    ts = ziptuples(M.manifolds, submanifold_components(M, p), submanifold_components(M, X))
    for t ∈ ts
        err = check_tangent_vector(t...; kwargs...)
        err === nothing || return err
    end
    return nothing
end

@doc raw"""
    cross(M,N)
    cross(M1, M2, M3,...)

Return the [`ProductManifold`](@ref) For two [`Manifold`](@ref)s `M` and `N`,
where for the case that one of them is a [`ProductManifold`](@ref) itself,
the other is either prepended (if `N` is a product) or appenden (if `M`) is.
If both are product manifold, they are combined into one product manifold,
keeping the order.

For the case that more than one is a product manifold of these is build with the
same approach as above
"""
cross(::Manifold...)
cross(M1::Manifold, M2::Manifold) = ProductManifold(M1, M2)
cross(M1::ProductManifold, M2::Manifold) = ProductManifold(M1.manifolds..., M2)
cross(M1::Manifold, M2::ProductManifold) = ProductManifold(M1, M2.manifolds...)
function cross(M1::ProductManifold, M2::ProductManifold)
    return ProductManifold(M1.manifolds..., M2.manifolds...)
end

function det_local_metric(M::MetricManifold{ProductManifold,ProductMetric}, p::ProductArray)
    dets = map(det_local_metric, M.manifolds, submanifold_components(M, p))
    return prod(dets)
end

@doc raw"""
    distance(M::ProductManifold, p, q)

Compute the distance between two points `p` and `q` on the [`ProductManifold`](@ref) `M`, which is
the 2-norm of the elementwise distances on the internal manifolds that build `M`.
"""
function distance(M::ProductManifold, p, q)
    return sqrt(sum(
        map(
            distance,
            M.manifolds,
            submanifold_components(M, p),
            submanifold_components(M, q),
        ) .^ 2,
    ))
end

@doc raw"""
    exp(M::ProductManifold, p, X)

compute the exponential map from `p` in the direction of `X` on the [`ProductManifold`](@ref) `M`,
which is the elementwise exponential map on the internal manifolds that build `M`.
"""
exp(::ProductManifold, ::Any...)
function exp(M::ProductManifold, p::ProductRepr, X::ProductRepr)
    return ProductRepr(map(
        exp,
        M.manifolds,
        submanifold_components(M, p),
        submanifold_components(M, X),
    )...)
end

function exp!(M::ProductManifold, q, p, X)
    map(
        exp!,
        M.manifolds,
        submanifold_components(M, q),
        submanifold_components(M, p),
        submanifold_components(M, X),
    )
    return q
end

@doc raw"""
    flat(M::ProductManifold, p, w::FVector{TangentSpaceType})

use the musical isomorphism to transform the tangent vector `w` from the tangent space at
`x` on the [`ProductManifold`](@ref) `M` to a cotangent vector.
This can be done elementwise for every entry of `w` (and `x`) separately.
"""
flat(::ProductManifold, ::Any...)

function flat!(M::ProductManifold, ξ::CoTFVector, p, X::TFVector)
    vfs = map(u -> FVector(CotangentSpace, u), submanifold_components(ξ))
    wfs = map(u -> FVector(TangentSpace, u), submanifold_components(X))
    map(flat!, M.manifolds, vfs, submanifold_components(M, p), wfs)
    return ξ
end

function get_basis(M::ProductManifold, p, B::AbstractBasis)
    parts = map(t -> get_basis(t..., B), ziptuples(M.manifolds, submanifold_components(p)))
    return PrecomputedProductOrthonormalBasis(parts)
end
function get_basis(M::ProductManifold, p, B::DiagonalizingOrthonormalBasis)
    vs = map(ziptuples(
        M.manifolds,
        submanifold_components(p),
        submanifold_components(B.v),
    )) do t
        return get_basis(t[1], t[2], DiagonalizingOrthonormalBasis(t[3]))
    end
    return PrecomputedProductOrthonormalBasis(vs)
end
function get_basis(M::ProductManifold, p, B::ArbitraryOrthonormalBasis)
    parts = map(t -> get_basis(t..., B), ziptuples(M.manifolds, submanifold_components(p)))
    return PrecomputedProductOrthonormalBasis(parts)
end

function get_coordinates(M::ProductManifold, p, X, B::PrecomputedProductOrthonormalBasis)
    reps = map(
        get_coordinates,
        M.manifolds,
        submanifold_components(p),
        submanifold_components(X),
        B.parts,
    )
    return vcat(reps...)
end
function get_coordinates(M::ProductManifold, p, X, B::ArbitraryOrthonormalBasis)
    reps = map(
        t -> get_coordinates(t..., B),
        ziptuples(M.manifolds, submanifold_components(p), submanifold_components(X)),
    )
    return vcat(reps...)
end

function get_vector(
    M::ProductManifold{<:NTuple{N,Any}},
    p::ProductRepr,
    X,
    B::PrecomputedProductOrthonormalBasis,
) where {N}
    dims = map(manifold_dimension, M.manifolds)
    dims_acc = accumulate(+, [1, dims...])
    parts = ntuple(N) do i
        get_vector(
            M.manifolds[i],
            submanifold_component(p, i),
            X[dims_acc[i]:dims_acc[i]+dims[i]-1],
            B.parts[i],
        )
    end
    return ProductRepr(parts)
end
function get_vector(
    M::ProductManifold{<:NTuple{N,Any}},
    p::ProductRepr,
    X,
    B::ArbitraryOrthonormalBasis,
) where {N}
    dims = map(manifold_dimension, M.manifolds)
    dims_acc = accumulate(+, [1, dims...])
    parts = ntuple(N) do i
        get_vector(
            M.manifolds[i],
            submanifold_component(p, i),
            X[dims_acc[i]:dims_acc[i]+dims[i]-1],
            B,
        )
    end
    return ProductRepr(parts)
end

function get_vectors(
    M::ProductManifold{<:NTuple{N,Manifold}},
    p::ProductRepr,
    B::PrecomputedProductOrthonormalBasis,
) where {N}
    xparts = submanifold_components(p)
    BVs = map(t -> get_vectors(t...), ziptuples(M.manifolds, xparts, B.parts))
    zero_tvs = map(t -> zero_tangent_vector(t...), ziptuples(M.manifolds, xparts))
    vs = typeof(ProductRepr(zero_tvs...))[]
    for i = 1:N, k = 1:length(BVs[i])
        push!(vs, ProductRepr(zero_tvs[1:i-1]..., BVs[i][k], zero_tvs[i+1:end]...))
    end
    return vs
end

function hat!(M::ProductManifold, X, p, Xⁱ)
    dim = manifold_dimension(M)
    @assert length(Xⁱ) == dim
    i = one(dim)
    ts = ziptuples(M.manifolds, submanifold_components(M, X), submanifold_components(M, p))
    for t ∈ ts
        dim = manifold_dimension(first(t))
        tvⁱ = @inbounds view(Xⁱ, i:(i+dim-1))
        hat!(t..., tvⁱ)
        i += dim
    end
    return X
end

@doc raw"""
    injectivity_radius(M::ProductManifold)
    injectivity_radius(M::ProductManifold, x)

Compute the injectivity radius on the [`ProductManifold`](@ref), which is the
minimum of the factor manifolds.
"""
injectivity_radius(::ProductManifold, ::Any...)
function injectivity_radius(M::ProductManifold, p)
    return min(map(injectivity_radius, M.manifolds, submanifold_components(M, p))...)
end
injectivity_radius(M::ProductManifold) = min(map(injectivity_radius, M.manifolds)...)

@doc raw"""
    inner(M::ProductManifold, p, X, Y)

compute the inner product of two tangent vectors `X`, `Y` from the tangent space
at `p` on the [`ProductManifold`](@ref) `M`, which is just the sum of the
internal manifolds that build `M`.
"""
function inner(M::ProductManifold, p, X, Y)
    subproducts = map(
        inner,
        M.manifolds,
        submanifold_components(M, p),
        submanifold_components(M, X),
        submanifold_components(M, Y),
    )
    return sum(subproducts)
end

@doc raw"""
    inverse_retract(M::ProductManifold, p, q, m::InverseProductRetraction)

Compute the inverse retraction from `p` with respect to `q` on the [`ProductManifold`](@ref)
`M` using an [`InverseProductRetraction`](@ref), which by default encapsulates a inverse
retraction for each manifold of the product. Then this method is performed elementwise,
so the encapsulated inverse retraction methods have to be available per factor.
"""
inverse_retract(::ProductManifold, ::Any, ::Any, ::Any, ::InverseProductRetraction)

function inverse_retract!(M::ProductManifold, X, p, q, method::InverseProductRetraction)
    map(
        inverse_retract!,
        M.manifolds,
        submanifold_components(M, X),
        submanifold_components(M, p),
        submanifold_components(M, q),
        method.inverse_retractions,
    )
    return X
end

is_default_metric(::ProductManifold, ::ProductMetric) = Val(true)

function isapprox(M::ProductManifold, p, q; kwargs...)
    return all(
        t -> isapprox(t...; kwargs...),
        ziptuples(M.manifolds, submanifold_components(M, p), submanifold_components(M, q)),
    )
end
function isapprox(M::ProductManifold, p, X, Y; kwargs...)
    return all(
        t -> isapprox(t...; kwargs...),
        ziptuples(
            M.manifolds,
            submanifold_components(M, p),
            submanifold_components(M, X),
            submanifold_components(M, Y),
        ),
    )
end

@doc raw"""
    log(M::ProductManifold, p, q)

Compute the logarithmic map from `p` to `q` on the [`ProductManifold`](@ref) `M`,
which can be computed using the logarithmic maps of the manifolds elementwise.
"""
log(::ProductManifold, ::Any...)
function log(M::ProductManifold, p::ProductRepr, q::ProductRepr)
    return ProductRepr(map(
        log,
        M.manifolds,
        submanifold_components(M, p),
        submanifold_components(M, q),
    )...)
end

function log!(M::ProductManifold, X, p, q)
    map(
        log!,
        M.manifolds,
        submanifold_components(M, X),
        submanifold_components(M, p),
        submanifold_components(M, q),
    )
    return X
end

@doc raw"""
    manifold_dimension(M::ProductManifold)

Return the manifold dimension of the [`ProductManifold`](@ref), which is the sum of the
manifold dimensions the product is made of.
"""
manifold_dimension(M::ProductManifold) = mapreduce(manifold_dimension, +, M.manifolds)

@doc raw"""
    norm(M::PowerManifold, p, X)

Compute the norm of `X` from the tangent space of `p` on the [`ProductManifold`](@ref),
i.e. from the element wise norms the 2-norm is computed.
"""
function norm(M::ProductManifold, p, X)
    norms_squared = (
        map(
            norm,
            M.manifolds,
            submanifold_components(M, p),
            submanifold_components(M, X),
        ) .^ 2
    )
    return sqrt(sum(norms_squared))
end

function ProductFVectorDistribution(
    type::VectorBundleFibers{<:VectorSpaceType,<:ProductManifold},
    p::Union{AbstractArray,MPoint,ProductRepr},
    distributions::FVectorDistribution...,
)
    return ProductFVectorDistribution{typeof(type),typeof(distributions),typeof(p)}(
        type,
        p,
        distributions,
    )
end
function ProductFVectorDistribution(
    type::VectorBundleFibers{<:VectorSpaceType,<:ProductManifold},
    distributions::FVectorDistribution...,
)
    p = ProductRepr(map(d -> support(d).point, distributions))
    return ProductFVectorDistribution(type, p, distributions...)
end
function ProductFVectorDistribution(distributions::FVectorDistribution...)
    M = ProductManifold(map(d -> support(d).space.manifold, distributions)...)
    fiber = support(distributions[1]).space.fiber
    if !all(d -> support(d).space.fiber == fiber, distributions)
        error("Not all distributions have support in vector spaces of the same type, which is currently not supported")
    end
    # Probably worth considering sum spaces in the future?
    x = ProductRepr(map(d -> support(d).point, distributions)...)
    return ProductFVectorDistribution(VectorBundleFibers(fiber, M), x, distributions...)
end

function ProductPointDistribution(M::ProductManifold, distributions::MPointDistribution...)
    return ProductPointDistribution{typeof(M),typeof(distributions)}(M, distributions)
end
function ProductPointDistribution(distributions::MPointDistribution...)
    M = ProductManifold(map(d -> support(d).manifold, distributions)...)
    return ProductPointDistribution(M, distributions...)
end

function rand(rng::AbstractRNG, d::ProductPointDistribution)
    return ProductRepr(map(d -> rand(rng, d), d.distributions)...)
end
function rand(rng::AbstractRNG, d::ProductFVectorDistribution)
    return ProductRepr(map(d -> rand(rng, d), d.distributions)...)
end

function _rand!(rng::AbstractRNG, d::ProductPointDistribution, x::AbstractArray{<:Number})
    return copyto!(x, rand(rng, d))
end
function _rand!(rng::AbstractRNG, d::ProductPointDistribution, p::ProductRepr)
    map(
        t -> _rand!(rng, t[1], t[2]),
        d.distributions,
        submanifold_components(d.manifold, p),
    )
    return p
end
function _rand!(rng::AbstractRNG, d::ProductFVectorDistribution, v::AbstractArray{<:Number})
    return copyto!(v, rand(rng, d))
end
function _rand!(rng::AbstractRNG, d::ProductFVectorDistribution, X::ProductRepr)
    map(t -> _rand!(rng, t[1], t[2]), d.distributions, submanifold_components(d.space.manifold, X))
    return X
end

@doc raw"""
    retract(M::ProductManifold, p, X, m::ProductRetraction)

Compute the retraction from `p` with tangent vector `X` on the [`ProductManifold`](@ref) `M`
using an [`ProductRetraction`](@ref), which by default encapsulates retractions of the
base manifolds. Then this method is performed elementwise, so the encapsulated retractions
method has to be one that is available on the manifolds.
"""
retract(::ProductManifold, ::Any...)

function retract!(M::ProductManifold, q, p, X, method::ProductRetraction)
    map(
        retract!,
        M.manifolds,
        submanifold_components(M, q),
        submanifold_components(M, p),
        submanifold_components(M, X),
        method.retractions,
    )
    return q
end

function representation_size(M::ProductManifold)
    return (mapreduce(m -> prod(representation_size(m)), +, M.manifolds),)
end

@doc raw"""
    sharp(M::ProductManifold, p, ξ::FVector{CotangentSpaceType})

Use the musical isomorphism to transform the cotangent vector `ξ` from the tangent space at
`p` on the [`ProductManifold`](@ref) `M` to a tangent vector.
This can be done elementwise, so vor every entry of `ξ` (and `p`) sparately
"""
sharp(::ProductManifold, ::Any...)

function sharp!(M::ProductManifold, X::TFVector, p, ξ::CoTFVector)
    vfs = map(u -> FVector(TangentSpace, u), submanifold_components(X))
    wfs = map(u -> FVector(CotangentSpace, u), submanifold_components(ξ))
    map(sharp!, M.manifolds, vfs, submanifold_components(M, p), wfs)
    return X
end

function _show_submanifold(io::IO, M::Manifold; pre = "")
    sx = sprint(show, "text/plain", M, context = io, sizehint = 0)
    if occursin('\n', sx)
        sx = sprint(show, M, context = io, sizehint = 0)
    end
    sx = replace(sx, '\n' => "\n$(pre)")
    print(io, pre, sx)
end

function _show_submanifold_range(io::IO, Ms, range; pre = "")
    for i in range
        M = Ms[i]
        print(io, '\n')
        _show_submanifold(io, M; pre = pre)
    end
end

function _show_product_manifold_no_header(io::IO, M)
    n = length(M.manifolds)
    sz = displaysize(io)
    screen_height, screen_width = sz[1] - 4, sz[2]
    half_height = div(screen_height, 2)
    inds = 1:n
    pre = " "
    if n > screen_height
        inds = [1:half_height; (n-div(screen_height - 1, 2)+1):n]
    end
    if n ≤ screen_height
        _show_submanifold_range(io, M.manifolds, 1:n; pre = pre)
    else
        _show_submanifold_range(io, M.manifolds, 1:half_height; pre = pre)
        print(io, "\n$(pre)⋮")
        _show_submanifold_range(
            io,
            M.manifolds,
            (n-div(screen_height - 1, 2)+1):n;
            pre = pre,
        )
    end
    return nothing
end

function show(io::IO, mime::MIME"text/plain", M::ProductManifold)
    n = length(M.manifolds)
    print(io, "ProductManifold with $(n) submanifold$(n == 1 ? "" : "s"):")
    _show_product_manifold_no_header(io, M)
end

function show(io::IO, M::ProductManifold)
    print(io, "ProductManifold(", join(M.manifolds, ", "), ")")
end

"""
    submanifold(M::ProductManifold, i::Integer)

Extract the `i`th factor of the product manifold `M`.
"""
submanifold(M::ProductManifold, i::Integer) = M.manifolds[i]

"""
    submanifold(M::ProductManifold, i::Val)
    submanifold(M::ProductManifold, i::AbstractVector)

Extract the factor of the product manifold `M` indicated by indices in `i`.
For example, for `i` equal to `Val((1, 3))` the product manifold constructed
from the first and the third factor is returned.

The version with `AbstractVector` is not type-stable, for better preformance use `Val`.
"""
submanifold(M::ProductManifold, i::Val) = ProductManifold(select_from_tuple(M.manifolds, i))
submanifold(M::ProductManifold, i::AbstractVector) = submanifold(M, Val(tuple(i...)))

support(d::ProductPointDistribution) = MPointSupport(d.manifold)
function support(tvd::ProductFVectorDistribution)
    return FVectorSupport(
        tvd.type,
        ProductRepr(map(d -> support(d).point, tvd.distributions)...),
    )
end

function vee!(M::ProductManifold, Xⁱ, p, X)
    dim = manifold_dimension(M)
    @assert length(Xⁱ) == dim
    i = one(dim)
    ts = ziptuples(M.manifolds, submanifold_components(M, p), submanifold_components(M, X))
    for t ∈ ts
        SM = first(t)
        dim = manifold_dimension(SM)
        tvⁱ = @inbounds view(Xⁱ, i:(i+dim-1))
        vee!(SM, tvⁱ, Base.tail(t)...)
        i += dim
    end
    return Xⁱ
end

function zero_tangent_vector!(M::ProductManifold, X, p)
    map(
        zero_tangent_vector!,
        M.manifolds,
        submanifold_components(M, X),
        submanifold_components(M, p),
    )
    return X
end
