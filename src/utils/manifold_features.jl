#
# Towards more automated testing – let's find Methods available on a manifold automatically
# but due to decorators, we have to test_try_calls
#
@doc raw"""
    find_manifold_features(M; kwargs...)

Find functions that are available on a manifold by trying to call them

Returns a vector of functions.
Note that `retract`, `inverse_retract` and `vector_transport` are only tested
if a default is available for exactly that default. their more precise versions are
[`find_retractions`](@ref), [`find_inverse_retractions`](@ref), [`find_vector_transports`](@ref) (still todo)

Note that this only includes the high-level / exported functions and only their allocating variants.

# Keyword Arguments

The methods might depend on which types of points, tangent vectors or numbers are chosen,
these can be set by

* `p`     – (`rand(M)`) a point
* `X`     - (`rand(M; vector_at=p)`)` a tangent vector
* `t`     – (`1.0`) a value
"""
function find_manifold_features(M; p=rand(M), X=rand(M; vector_at=p), t=1.0)
    # This is a bit of an exhaustive list of try-catch checks, since we just try whether something does not fail
    features = Function[]
    #
    # a) signature f(M)
    for f in [injectivity_radius, manifold_dimension, rand]
        try
            f(M, p, p)
            push!(features, f)
        catch
        end
    end
    # b) signature f(M, p)
    for f in [is_point, embed, zero_vector]
        try
            f(M, p)
            push!(features, f)
        catch
        end
    end
    # c) signature f(M, p, X)
    for f in [exp, retract, is_vector]
        try
            f(M, p, X)
            push!(features, f)
        catch
        end
    end
    # d) signatures f(M, p, q)
    for f in [log, inverse_retract, distance]
        try
            f(M, p, p)
            push!(features, f)
        catch
        end
    end
    # e) signature f(M, p, X, q)
    for f in [parallel_transport_to, vector_transport_to]
        try
            f(M, p, X, p)
            push!(features, f)
        catch
        end
    end
    # f) signature f(M, p, X, Y)
    for f in [parallel_transport_direction, vector_transport_direction]
        try
            f(M, p, X, X)
            push!(features, f)
        catch
        end
    end
    return features
end

"""
    ManifoldFeatures

Collect a set of features available on a manifold.

# Fields

* `functions`
* `retractions`
* `inverseretractions`
* `vector_transports`
* `tolerances` – based on functions or symbols provide a tolerance
"""
struct ManifoldFeatures{F<:Function}
    functions::Vector{F}
    retractions::Vector{AbstractRetractionMethod}
    inverse_retractions::Vector{AbstractRetractionMethod}
    vector_transports::Vector{AbstractRetractionMethod}
    tolerances::Dict{Union{Function,Symbol},Float64}
end
