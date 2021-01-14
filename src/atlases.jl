
"""
    abstract type AbstractAtlas end

An abstract class for atlases.
"""
abstract type AbstractAtlas end

"""
    struct RetractionAtlas{
        TInvRetr<:AbstractInverseRetractionMethod,
        TRetr<:AbstractRetractionMethod,
        TBasis<:AbstractBasis,
    } <: AbstractAtlas

An atlas indexed by points on a manifold, such that coordinate transformations are performed
using retractions, inverse retractions and coordinate calculation for a given basis.
"""
struct RetractionAtlas{
    TInvRetr<:AbstractInverseRetractionMethod,
    TRetr<:AbstractRetractionMethod,
    TBasis<:AbstractBasis,
} <: AbstractAtlas
    invretr::TInvRetr
    retr::TRetr
    basis::TBasis
end

function RetractionAtlas(
    invretr::AbstractInverseRetractionMethod,
    retr::AbstractRetractionMethod,
)
    return RetractionAtlas(invretr, retr, DefaultOrthonormalBasis())
end

function get_default_atlas(M::Manifold)
    return RetractionAtlas(LogarithmicInverseRetraction(), ExponentialRetraction())
end

"""
    get_point_coordinates(M::Manifold, A::AbstractAtlas, i, p)

Calculate coordinates of point `p` on manifold `M` in chart from atlas `A` at index `i`.
"""
get_point_coordinates(::Manifold, ::AbstractAtlas, ::Any, ::Any)

function get_point_coordinates(M::Manifold, A::AbstractAtlas, i, p)
    x = allocate_result(M, get_point_coordinates, p)
    get_point_coordinates!(M, x, A, i, p)
    return x
end

function get_point_coordinates!(M::Manifold, x, A::RetractionAtlas, i, p)
    return get_coordinates!(M, x, i, inverse_retract(M, i, p, A.invretr), A.basis)
end

function get_point_coordinates(M::Manifold, A::RetractionAtlas, i, p)
    return get_coordinates(M, i, inverse_retract(M, i, p, A.invretr), A.basis)
end

"""
    get_point(M::Manifold, A::AbstractAtlas, i, x)

Calculate point at coordinates `x` on manifold `M` in chart from atlas `A` at index `i`.
"""
get_point(::Manifold, ::AbstractAtlas, ::Any, ::Any)

function get_point(M::Manifold, A::RetractionAtlas, i, x)
    return retract(M, i, get_vector(M, i, x, A.basis), A.retr)
end

function get_point!(M::Manifold, p, A::RetractionAtlas, i, x)
    return retract!(M, p, i, get_vector(M, i, x, A.basis), A.retr)
end

"""
    select_chart(M::Manifold, A::AbstractAtlas, p)

Select a chart from atlas `A` for manifold `M` that is suitable for representing
neighborhood of point `p`.
"""
select_chart(::Manifold, ::AbstractAtlas, ::Any)

select_chart(::Manifold, ::RetractionAtlas, p) = p

"""
    transition_map(M::Manifold, A_from::AbstractAtlas, i_from, A_to::AbstractAtlas, i_to, x)
    transition_map(M::Manifold, A::AbstractAtlas, i_from, i_to, x)

Given coordinates `x` in chart `(A_from, i_from)` of a point on manifold `M`, returns
coordinates of that point in chart `(A_to, i_to)`. If `A_from` and `A_to` are equal, `A_to`
can be omitted.
"""
function transition_map(
    M::Manifold,
    A_from::AbstractAtlas,
    i_from,
    A_to::AbstractAtlas,
    i_to,
    x,
)
    return get_point_coordinates(M, A_to, i_to, get_point(M, A_from, i_from, x))
end

function transition_map(M::Manifold, A::AbstractAtlas, i_from, i_to, x)
    return transition_map(M, A, i_from, A, i_to, x)
end

function transition_map!(
    M::Manifold,
    y,
    A_from::AbstractAtlas,
    i_from,
    A_to::AbstractAtlas,
    i_to,
    x,
)
    return get_point_coordinates!(M, y, A_to, i_to, get_point(M, A_from, i_from, x))
end

function transition_map!(M::Manifold, y, A::AbstractAtlas, i_from, i_to, x)
    return transition_map!(M, y, A, i_from, A, i_to, x)
end
