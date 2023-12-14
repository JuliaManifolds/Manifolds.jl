function _vector_transport_direction(
    M::VectorBundle,
    p,
    X,
    d,
    m::FiberBundleProductVectorTransport,
)
    px, pVx = submanifold_components(M.manifold, p)
    VXM, VXF = submanifold_components(M.manifold, X)
    dx, dVx = submanifold_components(M.manifold, d)
    return ArrayPartition(
        vector_transport_direction(M.manifold, px, VXM, dx, m.method_horizontal),
        bundle_transport_tangent_direction(M, px, pVx, VXF, dx, m.method_vertical),
    )
end

function _vector_transport_to(
    M::VectorBundle,
    p,
    X,
    q,
    m::FiberBundleProductVectorTransport,
)
    px, pVx = submanifold_components(M.manifold, p)
    VXM, VXF = submanifold_components(M.manifold, X)
    qx, qVx = submanifold_components(M.manifold, q)
    return ArrayPartition(
        vector_transport_to(M.manifold, px, VXM, qx, m.method_horizontal),
        bundle_transport_tangent_to(M, px, pVx, VXF, qx, m.method_vertical),
    )
end