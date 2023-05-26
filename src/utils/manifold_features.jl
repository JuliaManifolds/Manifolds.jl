#
# Towards more automated testing – let's find Methods available on a manifold automatically
# but due to decorators, we have to test_try_calls
#
@doc raw"""
    find_manifold_functions(M; kwargs...)

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
function find_manifold_functions(M; p=rand(M), X=rand(M; vector_at=p), t=1.0)
    # This is a bit of an exhaustive list of try-catch checks, since we just try whether something does not fail
    features = Function[]
    #
    # a) signature f(M)
    for f in [injectivity_radius, manifold_dimension, rand, representation_size]
        try
            f(M)
            push!(features, f)
        catch
        end
    end
    # b) signature f(M, p)
    for f in [copy, is_point, embed, zero_vector]
        try
            f(M, p)
            push!(features, f)
        catch
        end
    end
    # c) signature f(M, p, X)
    for f in [exp, geodesic, is_vector, norm, retract]
        try
            f(M, p, X)
            push!(features, f)
        catch
        end
    end
    # d) signatures f(M, p, q)
    for f in [distance, inverse_retract, log, shortest_geodesic]
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
    # f) signature f(M, p, X, Y, Z)
    for f in [riemann_tensor]
        try
            f(M, p, X, X, X)
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
    inverse_retractions::Vector{AbstractInverseRetractionMethod}
    vector_transports::Vector{AbstractVectorTransportMethod}
end
function ManifoldFeatures(;
    functions::Vector{F}=Functions[],
    retractions=Vector{AbstractRetractionMethod}[],
    inverse_retractions=Vector{AbstractInverseRetractionMethod}[],
    vector_transports=Vector{AbstractVectorTransportMethod}[],
) where {F<:Function}
    return ManifoldFeatures{F}(
        functions,
        retractions,
        inverse_retractions,
        vector_transports,
    )
end
function ManifoldFeatures(
    M::AbstractManifold;
    functions::Vector{F}=find_manifold_functions(M),
    retractions=Vector{AbstractRetractionMethod}[],
    inverse_retractions=Vector{AbstractInverseRetractionMethod}[],
    vector_transports=Vector{AbstractVectorTransportMethod}[],
) where {F<:Function}
    return ManifoldFeatures{F}(
        functions,
        retractions,
        inverse_retractions,
        vector_transports,
    )
end

@doc """
    ManifoldExpectations

For the features from [`ManifoldFeatures`](@ref) this struct provides expected
values and tolerances

* `values`     a dictionary `:Symbol->value` for expected values, e.g. `:dimension`
* `strings`    a dictionary `:Symbol->String` for expected strings, e.g. `:repr` for `repr(M)`
* `tolerances` a dictionary `:Symbol->Float64` for tolerances in checks of functions.
"""
struct ManifoldExpectations
    values::Dict{Symbol,Int}
    strings::Dict{Symbol,String}
    tolerances::Dict{Symbol,Float64}
end
function ManifoldExpectations(;
    values=Dict{Symbol,Int}(),
    strings=Dict{Symbol,String}(),
    tolerances=Dict{Symbol,Int}(:default => 1e-14),
)
    return ManifoldExpectations(values, strings, tolerances)
end

@doc """
    has_feature_expectations(F::ManifoldFeatures, E::ManifoldExpectations, s::symbol)

For a current set of features and expectations infer whether they provide information
for a symbol `s`.

For example for `has_feature_expectations(F,E,:exp)` to return `true`
* the function `exp` has to be in the features
* the expectations have to have a tolerance present
"""
function has_feature_expectations(F::ManifoldFeatures, E::ManifoldExpectations, s::Symbol)
    # default cases
    f = getfield(Main, s) #default: is test a function name or a type?
    (f isa Function) && (f in F.functions) && (return true)
    (f isa AbstractRetractionMethod) && (f in F.retractions) && (return true)
    (f isa AbstractInverseRetractionMethod) && (f in F.retractions) && (return true)
    (f isa AbstractVectorTransportMethod) && (f in F.inverse_retractions) && (return true)
    return false
end
