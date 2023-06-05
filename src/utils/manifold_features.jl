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

function find_manifold_retractions(M; p=rand(M), X=rand(M; vector_at=p), t=1.0)
    checks = AbstractRetractionMethod[]
    # The following can only be checked on certain manifolds and/or need parameters
    auto_excl = [
        EmbeddedRetraction,
        RetractionWithKeywords,
        ODEExponentialRetraction,
        PadeRetraction,
        ProductRetraction, # generic on Products
        SasakiRetraction, # generic on tangent bundle
    ]
    for T in subtypes(AbstractRetractionMethod)
        if !isabstracttype(T) && T ∉ auto_excl
            push!(checks, T())
        end
    end
    push!(checks, PadeRetraction(2)) # Since order one might just fall back to Caley
    try #if we have an emebdding try embedded retraction with their default one
        push!(checks, EmbeddedRetraction(default_retraction_method(get_embedding(M))))
    catch
    end
    if default_retraction_method(M) != ExponentialRetraction()
        push!(checks, ODEExponentialRetraction(default_retraction_method(M)))
    end
    if M isa ProductManifold
        push!(
            checks,
            ProductRetraction([default_retraction_method(N) for N in M.manifolds]...),
        )
    end
    if M isa TangentBundle
        push!(checks, SasakiRetraction(1))
    end
    #
    # Ok – Let's check them
    retr_features = AbstractRetractionMethod[]
    for retr in checks
        try
            retract(M, p, X, retr)
            push!(retr_features, retr)
        catch
        end
    end
    return retr_features
end

function find_manifold_inverse_retractions(M; p=rand(M), X=rand(M; vector_at=p), t=1.0)
    checks = AbstractInverseRetractionMethod[]
    # The following can only be checked on certain manifolds and/or need parameters
    auto_excl = [
        EmbeddedInverseRetraction,
        PadeInverseRetraction,
        InverseProductRetraction,
        InverseRetractionWithKeywords,
    ]
    for T in subtypes(AbstractInverseRetractionMethod)
        if !isabstracttype(T) && T ∉ auto_excl
            push!(checks, T())
        end
    end
    push!(checks, PadeInverseRetraction(2)) # Since order one might just fall back to Caley
    try #if we have an emebdding try embedded retraction with their default one
        push!(
            checks,
            EmbeddedInverseRetraction(default_inverse_retraction_method(get_embedding(M))),
        )
    catch
    end
    if M isa ProductManifold
        push!(
            checks,
            InverseProductRetraction(
                [default_inverse_retraction_method(N) for N in M.manifolds]...,
            ),
        )
    end
    #
    # Ok – Let's check them
    inv_retr_features = AbstractInverseRetractionMethod[]
    for inv_retr in checks
        try
            inverse_retract(M, p, p, inv_retr)
            push!(inv_retr_features, inv_retr)
        catch
        end
    end
    return inv_retr_features
end

function find_manifold_vector_transports(M; p=rand(M), X=rand(M; vector_at=p), t=1.0)
    checks = AbstractVectorTransportMethod[]
    # The following can only be checked on certain manifolds and/or need parameters
    auto_excl = [
        ScaledVectorTransport,
        VectorTransportWithKeywords,
        DifferentiatedRetractionVectorTransport,
        VectorTransportTo, #only generic for internal use
    ]
    for T in [
        subtypes(AbstractVectorTransportMethod)...,
        subtypes(AbstractLinearVectorTransportMethod)...,
    ]
        if !isabstracttype(T) && T ∉ auto_excl
            push!(checks, T())
        end
    end
    try
        push!(checks, DifferentiatedRetractionVectorTransport(default_retraction_method(M)))
    catch
    end
    try #if we have an emebdding try embedded retraction with their default one
        push!(checks, ScaledVectorTransport(default_vector_transport_method(M)))
    catch
    end
    #
    # Ok – Let's check them
    vector_transport_features = AbstractVectorTransportMethod[]
    for vector_transport in checks
        try
            vector_transport_to(M, p, X, p, vector_transport)
            push!(vector_transport_features, vector_transport)
        catch
        end
    end
    return vector_transport_features
end
"""
    ManifoldFeatures

Collect a set of features available on a manifold.

# Fields

* `functions`
* `retractions`
* `inverseretractions`
* `vector_transports`
* `properties` – specify further properties to be checked (traits or decorators)

All these are filled by default calling the corresponding `find_manifold_` functions,
though the properties variant is still under development
"""
struct ManifoldFeatures{F<:Function}
    functions::Vector{F}
    retractions::Vector{AbstractRetractionMethod}
    inverse_retractions::Vector{AbstractInverseRetractionMethod}
    vector_transports::Vector{AbstractVectorTransportMethod}
    properties::Dict{Symbol,Bool}
end
function ManifoldFeatures(;
    functions::Vector{F}=Functions[],
    retractions=Vector{AbstractRetractionMethod}[],
    inverse_retractions=Vector{AbstractInverseRetractionMethod}[],
    vector_transports=Vector{AbstractVectorTransportMethod}[],
    properties=Dict{Symbol,Bools}(),
) where {F<:Function}
    return ManifoldFeatures{F}(
        functions,
        retractions,
        inverse_retractions,
        vector_transports,
        properties,
    )
end
function ManifoldFeatures(
    M::AbstractManifold;
    p=rand(M),
    X=rand(M; vector_at=p),
    t=1.0,
    functions::Vector{F}=find_manifold_functions(M; p=p, X=X, t=t),
    retractions::Vector{R}=find_manifold_retractions(M; p=p, X=X, t=t),
    inverse_retractions::Vector{I}=find_manifold_inverse_retractions(M; p=p, X=X, t=t),
    vector_transports::Vector{V}=find_manifold_vector_transports(M; p=p, X=X, t=t),
    properties=Dict{Symbol,Bool}(),
) where {
    F<:Function,
    R<:AbstractRetractionMethod,
    I<:AbstractInverseRetractionMethod,
    V<:AbstractVectorTransportMethod,
}
    return ManifoldFeatures{F}(
        functions,
        retractions,
        inverse_retractions,
        vector_transports,
        properties,
    )
end
function show(io::IO, mf::ManifoldFeatures)
    # Print features to terminal
    s = """
    ManifoldFeatures\n\n

    Functions
    $(join(["  * $(f)" for f in mf.functions],"\n"))

    Retractions
    $(join(["  * $(typeof(r))" for r in mf.retractions],"\n"))

    Inverse Retractions
    $(join(["  * $(typeof(ir))" for ir in mf.inverse_retractions],"\n"))

    Vector transports
    $(join(["  * $(typeof(v))" for v in mf.vector_transports],"\n"))
    """
    return print(io, s)
end

function show(io::IO, ::MIME"test/markdown", mf::ManifoldFeatures)
    # Print the features in a nice Markdown table for the docs
end
@doc """
    ManifoldExpectations

For the features from [`ManifoldFeatures`](@ref) this struct provides expected
values and tolerances

* `values`     a dictionary `:Symbol->value` for expected values, e.g. `:dimension`
* `strings`    a dictionary `:Symbol->String` for expected strings, e.g. `:repr` for `repr(M)`
* `tolerances` a dictionary `:Symbol->Float64` for tolerances in checks of functions.
"""
struct ManifoldExpectations{V}
    values::Dict{Symbol,V}
    tolerances::Dict{Symbol,Float64}
end
function ManifoldExpectations(;
    values::Dict{Symbol,V}=Dict{Symbol,Float64}(),
    tolerances=Dict(:default => 1e-14),
) where {V}
    return ManifoldExpectations{V}(values, tolerances)
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
