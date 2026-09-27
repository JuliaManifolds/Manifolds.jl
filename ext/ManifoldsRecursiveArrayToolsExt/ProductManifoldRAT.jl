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

function allocate_result(M::ProductManifold, f::typeof(get_point), a::AbstractVector)
    return ArrayPartition(map(M_i -> allocate_result(M_i, f, a), M.manifolds)...)
end

function get_coordinates(
        M::ProductManifold,
        p,
        X,
        B::InducedBasis{ℝ, Manifolds.TangentSpaceType, <:ProductAtlas},
    )
    subbases = map((Mj, Aj, ij) -> induced_basis(Mj, Aj, ij), M.manifolds, B.A.atlases, B.i)
    reps = map(
        get_coordinates,
        M.manifolds,
        submanifold_components(M, p),
        submanifold_components(M, X),
        subbases,
    )
    return vcat(reps...)
end

function get_vector(
        M::ProductManifold,
        p::ArrayPartition,
        Xⁱ,
        B::InducedBasis{ℝ, Manifolds.TangentSpaceType, <:ProductAtlas},
    )
    dims = map(manifold_dimension, M.manifolds)
    @assert length(Xⁱ) == sum(dims)
    dim_ranges = ManifoldsBase._get_dim_ranges(dims)
    tXⁱ = map(dr -> (@inbounds view(Xⁱ, dr)), dim_ranges)
    subbases = map((Mj, Aj, ij) -> induced_basis(Mj, Aj, ij), M.manifolds, B.A.atlases, B.i)
    ts = Manifolds.ziptuples(M.manifolds, submanifold_components(M, p), tXⁱ, subbases)
    return ArrayPartition(map((@inline t -> get_vector(t...)), ts))
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

function riemannian_gradient(M::ProductManifold, p::ArrayPartition, X::ArrayPartition)
    return ArrayPartition(
        map(
            riemannian_gradient,
            M.manifolds,
            submanifold_components(M, p),
            submanifold_components(M, X),
        ),
    )
end
