
"""
    AbstractAtlas{𝔽}

An abstract class for atlases whith charts that have values in the vector space `𝔽ⁿ`
for some value of `n`. `𝔽` is a number system determined by an [`AbstractNumbers`](@ref)
object.
"""
abstract type AbstractAtlas{𝔽} end

"""
    RetractionAtlas{
        𝔽,
        TRetr<:AbstractRetractionMethod,
        TInvRetr<:AbstractInverseRetractionMethod,
        TBasis<:AbstractBasis,
    } <: AbstractAtlas{𝔽}

An atlas indexed by points on a manifold, such that coordinate transformations are performed
using retractions, inverse retractions, and coordinate calculation for a given basis.

# See also

[`AbstractAtlas`](@ref), [`AbstractInverseRetractionMethod`](@ref),
[`AbstractRetractionMethod`](@ref), [`AbstractBasis`](@ref)
"""
struct RetractionAtlas{
    𝔽,
    TRetr<:AbstractRetractionMethod,
    TInvRetr<:AbstractInverseRetractionMethod,
    TBasis<:AbstractBasis{<:Any,TangentSpaceType},
} <: AbstractAtlas{𝔽}
    retr::TRetr
    invretr::TInvRetr
    basis::TBasis
end

function RetractionAtlas(
    retr::AbstractRetractionMethod,
    invretr::AbstractInverseRetractionMethod,
)
    basis = DefaultOrthonormalBasis()
    return RetractionAtlas{ℝ,typeof(retr),typeof(invretr),typeof(basis)}(
        retr,
        invretr,
        basis,
    )
end
RetractionAtlas() = RetractionAtlas(ExponentialRetraction(), LogarithmicInverseRetraction())

"""
    get_default_atlas(::AbstractManifold)

Determine the default real-valued atlas for the given manifold.
"""
function get_default_atlas(::AbstractManifold)
    return RetractionAtlas()
end

"""
    get_point_coordinates(M::AbstractManifold, A::AbstractAtlas, i, p)

Calculate coordinates of point `p` on manifold `M` in chart from an [`AbstractAtlas`](@ref)
`A` at index `i`. Coordinates are in the number system determined by `A`.

# See also

[`get_point`](@ref), [`get_chart_index`](@ref)
"""
get_point_coordinates(::AbstractManifold, ::AbstractAtlas, ::Any, ::Any)

function get_point_coordinates(M::AbstractManifold, A::AbstractAtlas, i, p)
    a = allocate_result(M, get_point_coordinates, p)
    get_point_coordinates!(M, a, A, i, p)
    return a
end

function allocate_result(M::AbstractManifold, f::typeof(get_point_coordinates), p)
    T = allocate_result_type(M, f, (p,))
    return allocate(p, T, manifold_dimension(M))
end

function get_point_coordinates!(M::AbstractManifold, a, A::RetractionAtlas, i, p)
    return get_coordinates!(M, a, i, inverse_retract(M, i, p, A.invretr), A.basis)
end

function get_point_coordinates(M::AbstractManifold, A::RetractionAtlas, i, p)
    return get_coordinates(M, i, inverse_retract(M, i, p, A.invretr), A.basis)
end

"""
    get_point(M::AbstractManifold, A::AbstractAtlas, i, a)

Calculate point at coordinates `a` on manifold `M` in chart from an [`AbstractAtlas`](@ref)
`A` at index `i`.

# See also

[`get_point_coordinates`](@ref), [`get_chart_index`](@ref)
"""
get_point(::AbstractManifold, ::AbstractAtlas, ::Any, ::Any)

function get_point(M::AbstractManifold, A::AbstractAtlas, i, a)
    p = allocate_result(M, get_point, a)
    get_point!(M, p, A, i, a)
    return p
end

function allocate_result(M::AbstractManifold, f::typeof(get_point), a)
    T = allocate_result_type(M, f, (a,))
    return allocate(a, T, representation_size(M)...)
end

function get_point(M::AbstractManifold, A::RetractionAtlas, i, a)
    return retract(M, i, get_vector(M, i, a, A.basis), A.retr)
end

function get_point!(M::AbstractManifold, p, A::RetractionAtlas, i, a)
    return retract!(M, p, i, get_vector(M, i, a, A.basis), A.retr)
end

"""
    get_chart_index(M::AbstractManifold, A::AbstractAtlas, p)

Select a chart from an [`AbstractAtlas`](@ref) `A` for manifold `M` that is suitable for
representing the neighborhood of point `p`. This selection should be deterministic, although
different charts may be selected for arbitrarily close but distinct points.

# See also

[`get_default_atlas`](@ref)
"""
get_chart_index(::AbstractManifold, ::AbstractAtlas, ::Any)

get_chart_index(::AbstractManifold, ::RetractionAtlas, p) = p

"""
    transition_map(M::AbstractManifold, A_from::AbstractAtlas, i_from, A_to::AbstractAtlas, i_to, a)
    transition_map(M::AbstractManifold, A::AbstractAtlas, i_from, i_to, a)

Given coordinates `a` in chart `(A_from, i_from)` of a point on manifold `M`, returns
coordinates of that point in chart `(A_to, i_to)`. If `A_from` and `A_to` are equal, `A_to`
can be omitted.

# See also

[`AbstractAtlas`](@ref)
"""
function transition_map(
    M::AbstractManifold,
    A_from::AbstractAtlas,
    i_from,
    A_to::AbstractAtlas,
    i_to,
    a,
)
    return get_point_coordinates(M, A_to, i_to, get_point(M, A_from, i_from, a))
end

function transition_map(M::AbstractManifold, A::AbstractAtlas, i_from, i_to, a)
    return transition_map(M, A, i_from, A, i_to, a)
end

function transition_map!(
    M::AbstractManifold,
    y,
    A_from::AbstractAtlas,
    i_from,
    A_to::AbstractAtlas,
    i_to,
    a,
)
    return get_point_coordinates!(M, y, A_to, i_to, get_point(M, A_from, i_from, a))
end

function transition_map!(M::AbstractManifold, y, A::AbstractAtlas, i_from, i_to, a)
    return transition_map!(M, y, A, i_from, A, i_to, a)
end

"""
    induced_basis(M::AbstractManifold, A::AbstractAtlas, i, p, VST::VectorSpaceType)

Basis of vector space of type `VST` at point `p` from manifold `M` induced by
chart (`A`, `i`).

# See also

[`VectorSpaceType`](@ref), [`AbstractAtlas`](@ref)
"""
induced_basis(M::AbstractManifold, A::AbstractAtlas, i, VST::VectorSpaceType)

function induced_basis(
    ::AbstractManifold,
    A::RetractionAtlas{
        <:AbstractRetractionMethod,
        <:AbstractInverseRetractionMethod,
        <:DefaultOrthonormalBasis,
    },
    i,
    p,
    ::TangentSpaceType,
)
    return A.basis
end
function induced_basis(
    M::AbstractManifold,
    A::RetractionAtlas{
        <:AbstractRetractionMethod,
        <:AbstractInverseRetractionMethod,
        <:DefaultOrthonormalBasis,
    },
    i,
    p,
    ::CotangentSpaceType,
)
    return dual_basis(M, p, A.basis)
end

"""
    InducedBasis(vs::VectorSpaceType, A::AbstractAtlas, i)

The basis induced by chart with index `i` from an [`AbstractAtlas`](@ref) `A` of vector
space of type `vs`.

# See also

[`VectorSpaceType`](@ref), [`AbstractBasis`](@ref)
"""
struct InducedBasis{𝔽,VST<:VectorSpaceType,TA<:AbstractAtlas,TI} <: AbstractBasis{𝔽,VST}
    vs::VST
    A::TA
    i::TI
end

"""
    induced_basis(::AbstractManifold, A::AbstractAtlas, i, VST::VectorSpaceType)

Get the basis induced by chart with index `i` from an [`AbstractAtlas`](@ref) `A` of vector
space of type `vs`. Returns an object of type [`InducedBasis`](@ref).
"""
function induced_basis(
    ::AbstractManifold{𝔽},
    A::AbstractAtlas,
    i,
    VST::VectorSpaceType,
) where {𝔽}
    return InducedBasis{𝔽,typeof(VST),typeof(A),typeof(i)}(VST, A, i)
end

function dual_basis(
    M::AbstractManifold{𝔽},
    ::Any,
    B::InducedBasis{𝔽,TangentSpaceType},
) where {𝔽}
    return induced_basis(M, B.A, B.i, CotangentSpace)
end
function dual_basis(
    M::AbstractManifold{𝔽},
    ::Any,
    B::InducedBasis{𝔽,CotangentSpaceType},
) where {𝔽}
    return induced_basis(M, B.A, B.i, TangentSpace)
end

"""
    local_metric(M::AbstractManifold, p, B::InducedBasis)

Compute the local metric tensor for vectors expressed in terms of coordinates
in basis `B` on manifold `M`. The point `p` is not checked.
"""
local_metric(::AbstractManifold, ::Any, ::InducedBasis)
