
"""
    abstract type AbstractAtlas end

An abstract class for atlases.
"""
abstract type AbstractAtlas end

"""
    RetractionAtlas{TInvRetr<:AbstractInverseRetractionMethod,TRetr<:AbstractRetractionMethod,TBasis<:AbstractBasis} <: AbstractAtlas

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

function get_default_atlas(M::Manifold)
    return RetractionAtlas(LogarithmicInverseRetraction(), ExponentialRetraction())
end

"""
    select_chart(M::Manifold, A::AbstractAtlas, p)

Select a chart from atlas `A` for manifold `M` that is suitable for representing
neighborhood of point `p`.
"""
select_chart(::Manifold, ::AbstractAtlas, ::Any)

select_chart(::Manifold, ::RetractionAtlas, p) = p

"""
    get_point_coordinates(M::Manifold, A::AbstractAtlas, i, p)

Calculate coordinates of point `p` on manifold `M` in chart from atlas `A` at index `i`.
"""
get_point_coordinates(::Manifold, ::AbstractAtlas, ::Any, ::Any)

function get_point_coordinates(M::Manifold, A::RetractionAtlas, i, p)
    return get_coordinates(M, i, inverse_retract(M, i, p, A.invretr), A.basis)
end

"""
    get_point(M::Manifold, A::AbstractAtlas, i, x)

Calculate point at coordinates `x` on manifold `M` in chart from atlas `A` at index `i`.
"""
get_point(::Manifold, ::AbstractAtlas, ::Any, ::Any)

function get_point(M::Manifold, A::AbstractAtlas, i, x)
    return retract(M, i, get_vector(M, i, x, A.basis), A.retr)
end
