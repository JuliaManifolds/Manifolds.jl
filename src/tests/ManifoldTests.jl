
module ManifoldTests

using ..Manifolds
using ..ManifoldsBase

"""
    find_eps(x...)

Find an appropriate tolerance for given points or tangent vectors, or their types.
"""
find_eps(x...) = find_eps(Base.promote_type(map(number_eltype, x)...))
find_eps(x::Type{TN}) where {TN<:Number} = eps(real(TN))
find_eps(x) = find_eps(number_eltype(x))


function test_action end

function test_group end

function test_manifold end

function test_reversediff end

function test_forwarddiff end

export test_action, test_group, test_manifold

end # module
