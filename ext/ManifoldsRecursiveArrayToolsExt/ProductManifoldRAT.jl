
function adjoint_Jacobi_field(
    M::ProductManifold,
    p::ArrayPartition,
    q::ArrayPartition,
    t,
    X::ArrayPartition,
    β::Tβ,
) where {Tβ}
    return ArrayPartition(
        map(
            adjoint_Jacobi_field,
            M.manifolds,
            submanifold_components(M, p),
            submanifold_components(M, q),
            ntuple(_ -> t, length(M.manifolds)),
            submanifold_components(M, X),
            ntuple(_ -> β, length(M.manifolds)),
        )...,
    )
end

function jacobi_field(
    M::ProductManifold,
    p::ArrayPartition,
    q::ArrayPartition,
    t,
    X::ArrayPartition,
    β::Tβ,
) where {Tβ}
    return ArrayPartition(
        map(
            jacobi_field,
            M.manifolds,
            submanifold_components(M, p),
            submanifold_components(M, q),
            ntuple(_ -> t, length(M.manifolds)),
            submanifold_components(M, X),
            ntuple(_ -> β, length(M.manifolds)),
        )...,
    )
end
