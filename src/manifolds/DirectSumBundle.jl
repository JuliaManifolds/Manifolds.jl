
struct DirectSumType{TS<:Tuple} <: VectorSpaceType
    spaces::TS
end

DirectSumType(spaces::VectorSpaceType...) = DirectSumType{typeof(spaces)}(spaces)

function fiber_dimension(M::AbstractManifold, V::DirectSumType)
    dim = 0
    for space in V.spaces
        dim += fiber_dimension(M, space)
    end
    return dim
end

function vector_space_dimension(M::AbstractManifold, V::DirectSumType)
    dim = 0
    for space in V.spaces
        dim += vector_space_dimension(M, space)
    end
    return dim
end

const MultitangentBundle{N,𝔽,M} = VectorBundle{
    𝔽,
    DirectSumType{NTuple{N,TangentSpaceType}},
    M,
} where {𝔽,M<:AbstractManifold{𝔽}}

function MultitangentBundle{N}(M::AbstractManifold) where {N}
    return VectorBundle(DirectSumType(ntuple(f -> TangentSpace, N)), M)
end

const MultitangentBundleFibers{N,M} = VectorBundleFibers{
    DirectSumType{NTuple{N,TangentSpaceType}},
    M,
} where {M<:AbstractManifold}

function MultitangentBundleFibers{N}(M::AbstractManifold) where {N}
    return VectorBundleFibers(DirectSumType(ntuple(f -> TangentSpace, N)), M)
end

const MultitangentSpaceAtPoint{𝔽,N,M} = VectorSpaceAtPoint{
    𝔽,
    M,
    DirectSumType{NTuple{N,TangentSpaceType}},
} where {𝔽,M<:AbstractManifold{𝔽}}
function MultitangentSpaceAtPoint{N}(M::AbstractManifold, p) where {N}
    return VectorSpaceAtPoint(MultitangentBundleFibers{N}(M), p)
end

@inline function allocate_result(B::MultitangentBundleFibers{N}, f::TF) where {N,TF}
    return ArrayPartition(ntuple(_ -> allocate_result(B.manifold, f), Val(N))...)
end

function default_inverse_retraction_method(::MultitangentBundle)
    return FiberBundleInverseProductRetraction()
end

function default_retraction_method(::MultitangentBundle)
    return FiberBundleProductRetraction()
end

function default_vector_transport_method(B::MultitangentBundle)
    default_vt_m = default_vector_transport_method(B.manifold)
    return FiberBundleProductVectorTransport(default_vt_m, default_vt_m)
end

function get_basis(M::MultitangentBundleFibers, p, B::AbstractBasis{<:Any,TangentSpaceType})
    return get_basis(M.manifold, p, B)
end

function get_coordinates(M::MultitangentBundleFibers, p, X, B::AbstractBasis)
    return reduce(vcat, map(Xp -> get_coordinates(M.manifold, p, Xp, B), X.x))
end

function get_coordinates!(M::MultitangentBundleFibers, Y, p, X, B::AbstractBasis)
    Y .= reduce(vcat, map(Xp -> get_coordinates(M.manifold, p, Xp, B), X.x))
    return Y
end

function get_vector(M::MultitangentBundleFibers{N}, p, Xc, B::AbstractBasis) where {N}
    d = manifold_dimension(M.manifold)
    c_parts = ntuple(n -> view(Xc, (1 + ((n - 1) * d)):(n * d)), Val(N))
    return ArrayPartition(map(Xp -> get_vector(M.manifold, p, Xp, B), c_parts)...)
end

function get_vector!(M::MultitangentBundleFibers{N}, Y, p, Xc, B::AbstractBasis) where {N}
    d = manifold_dimension(M.manifold)
    c_parts = ntuple(n -> view(Xc, (1 + ((n - 1) * d)):(n * d)), Val(N))
    map((Yp, Xcp) -> get_vector!(M.manifold, Yp, p, Xcp, B), Y.x, c_parts)
    return Y
end

function get_vectors(M::MultitangentBundleFibers{N}, p, B::CachedBasis) where {N}
    bvs = get_vectors(M.manifold, p, B)
    X0 = zero_vector(M.manifold, p)
    vs = ArrayPartition{eltype(X0),NTuple{N,typeof(X0)}}[]
    for i in 1:N
        Xs1 = ntuple(_ -> X0, i - 1)
        Xs2 = ntuple(_ -> X0, N - i)
        for X in bvs
            push!(vs, ArrayPartition(Xs1..., X, Xs2...))
        end
    end
    return vs
end

function inner(M::MultitangentSpaceAtPoint, p, X, Y)
    return sum(map((Xp, Yp) -> inner(M.fiber.manifold, M.point, Xp, Yp), X.x, Y.x))
end
function inner(M::MultitangentBundleFibers, p, X, Y)
    return sum(map((Xp, Yp) -> inner(M.manifold, p, X, Y), X.x, Y.x))
end

function Random.rand!(rng::AbstractRNG, M::MultitangentSpaceAtPoint, X; vector_at=nothing)
    map(X.x) do Xp
        return rand!(rng, M.fiber.manifold, Xp; vector_at=M.point)
    end
    return X
end

function retract_product!(B::MultitangentBundle, q, p, X, t::Number)
    tX = t * X
    xp, Xps = submanifold_components(B.manifold, p)
    xq, Xqs = submanifold_components(B.manifold, q)
    VXM, VXFs = submanifold_components(B.manifold, tX)
    # this temporary avoids overwriting `p` when `q` and `p` occupy the same memory
    xqt = exp(B.manifold, xp, VXM)
    map(Xps.x, Xqs.x, VXFs.x) do Xp, Xq, VXF
        return vector_transport_direction!(
            B.manifold,
            Xq,
            xp,
            Xp + VXF,
            VXM,
            B.vector_transport.method_point,
        )
    end
    copyto!(B.manifold, xq, xqt)
    return q
end

function vector_transport_direction(
    M::MultitangentBundleFibers,
    p,
    X,
    d,
    m::AbstractVectorTransportMethod,
)
    return ArrayPartition(map(X.x) do VXF
        return vector_transport_direction(M.manifold, p, VXF, d, m)
    end)
end

function _vector_transport_direction!(
    M::MultitangentBundle,
    Y,
    p,
    X,
    d,
    m::FiberBundleProductVectorTransport,
)
    VYM, VYFs = submanifold_components(M.manifold, Y)
    px, pVxs = submanifold_components(M.manifold, p)
    VXM, VXFs = submanifold_components(M.manifold, X)
    dx, dVxs = submanifold_components(M.manifold, d)
    vector_transport_direction!(M.manifold, VYM, px, VXM, dx, m.method_point)
    map(VYFs.x, VXFs.x) do VYF, VXF
        return vector_transport_direction!(M.manifold, VYF, px, VXF, dx, m.method_fiber)
    end
    return Y
end

function _vector_transport_to(
    M::MultitangentBundle,
    p,
    X,
    q,
    m::FiberBundleProductVectorTransport,
)
    px, pVx = submanifold_components(M.manifold, p)
    VXM, VXFs = submanifold_components(M.manifold, X)
    qx, qVx = submanifold_components(M.manifold, q)
    vectors = map(VXFs.x) do VXF
        return vector_transport_to(M.manifold, px, VXF, qx, m.method_fiber)
    end
    return ArrayPartition(
        vector_transport_to(M.manifold, px, VXM, qx, m.method_point),
        ArrayPartition(vectors...),
    )
end

function vector_transport_to!(
    M::MultitangentBundle,
    Y,
    p,
    X,
    q,
    m::FiberBundleProductVectorTransport,
)
    px, pVx = submanifold_components(M.manifold, p)
    VXM, VXFs = submanifold_components(M.manifold, X)
    VYM, VYFs = submanifold_components(M.manifold, Y)
    qx, qVx = submanifold_components(M.manifold, q)
    vector_transport_to!(M.manifold, VYM, px, VXM, qx, m.method_point)
    map(VYFs.x, VXFs.x) do VYF, VXF
        return vector_transport_to!(M.manifold, VYF, px, VXF, qx, m.method_fiber)
    end
    return Y
end
function vector_transport_to!(
    M::MultitangentBundleFibers,
    Y,
    p,
    X,
    q,
    m::AbstractVectorTransportMethod,
)
    map(Y.x, X.x) do VYF, VXF
        return vector_transport_to!(M.manifold, VYF, p, VXF, q, m)
    end
    return Y
end

function zero_vector(B::MultitangentBundleFibers{N}, p) where {N}
    return ArrayPartition(ntuple(_ -> zero_vector(B.manifold, p), Val(N))...)
end

function zero_vector!(::MultitangentBundleFibers, X, p)
    return fill!(X, 0)
end
