
"""
    AbstractAffineConnection{𝔽,TM<:Manifold{𝔽}}

Affine connection on a manifold of type `TM`. Connection operation is defined using
[`apply_operator`](@ref).
"""
abstract type AbstractAffineConnection{𝔽,TM<:Manifold{𝔽}} end

@doc raw"""
    HessianOperator{TC<:AbstractAffineConnection}

Defines the Hessian operator for the given connection. Definition follows [^Absil2008],
Section 5.5. The formula reads:
````math
    \operatorname{Hess} f(p)[X_p] = \nabla_{X_p} \operatorname{grad} f
````

[^Absil2008]:
    >Absil, P.-A., Mahony, R., Sepulchre, R.:
    >Optimization Algorithms on Matrix Manifolds,
    > Princeton University Press, 2008,
    > doi: [10.1515/9781400830244](https://doi.org/10.1515/9781400830244),
    > [open access](http://press.princeton.edu/chapters/absil/).
"""
struct HessianOperator{TC<:AbstractAffineConnection}
    conn::TC
end

"""
    LeviCivitaConnection{𝔽,TM<:Manifold{𝔽}} <: AbstractAffineConnection{𝔽,TM}

Represents the Levi-Civita connection on manifold of type `TM`.
"""
struct LeviCivitaConnection{𝔽,TM<:Manifold{𝔽}} <: AbstractAffineConnection{𝔽,TM}
    manifold::TM
end


"""
    apply_operator(F::AbstractAffineConnection, p, X[, backend::AbstractRiemannianDiffBackend])

Apply connection `F` at point `p` to vector field `Y` in direction `X`.
"""
function apply_operator(
    F::AbstractAffineConnection,
    p,
    X,
    Y,
    backend::AbstractRiemannianDiffBackend,
)
    Z = allocate_result(base_manifold(F), apply_operator, p, X)
    return apply_operator!(F, Z, p, X, Y, backend)
end

function apply_operator(F::AbstractAffineConnection, p, X, Y)
    return apply_operator(F, p, X, Y, rdifferential_backend())
end


function apply_operator!(F::AbstractAffineConnection, Z, p, X, Y)
    return apply_operator!(F, Z, p, X, Y, rdifferential_backend())
end


base_manifold(F::LeviCivitaConnection) = F.manifold

@doc raw"""
    apply_operator(F::LeviCivitaConnection, p, X, Y)

Compute the value of the Levi-Civita connection at point `p`, in the direction pointed by
tangent vector `X` at `p`, of the vector field on `F.manifold` defined by a function `Y`.
The formula reads $(\nabla_X \mathit{Y})_p$.
"""
apply_operator(F::LeviCivitaConnection, p, X, Y)

"""
    apply_operator(
        F::LeviCivitaConnection{𝔽,<:AbstractEmbeddedManifold{DefaultIsometricEmbeddingType}},
        p,
        X,
        Y,
        backend::AbstractRiemannianDiffBackend,
    ) where {𝔽}

Apply the Levi-Civita connection on an isometrically embedded manifold by applying the
connection in the embedding and projecting it back.

See [^Absil2008], Section 5.3.3 for details.


[^Absil2008]:
    >Absil, P.-A., Mahony, R., Sepulchre, R.:
    >Optimization Algorithms on Matrix Manifolds,
    > Princeton University Press, 2008,
    > doi: [10.1515/9781400830244](https://doi.org/10.1515/9781400830244),
    > [open access](http://press.princeton.edu/chapters/absil/).
"""
function apply_operator(
    F::LeviCivitaConnection{𝔽,<:AbstractEmbeddedManifold{𝔽,DefaultIsometricEmbeddingType}},
    p,
    X,
    Y,
    backend::AbstractRiemannianDiffBackend,
) where {𝔽}
    emb_Z = apply_operator(
        LeviCivitaConnection(decorated_manifold(F.manifold)),
        embed(F.manifold, p),
        embed(F.manifold, p, X),
        q -> embed(F.manifold, q, Y(q)),
        backend,
     )
    return project(F.manifold, p, emb_Z)
end

function apply_operator!(
    F::LeviCivitaConnection{𝔽,<:AbstractEmbeddedManifold{𝔽,DefaultIsometricEmbeddingType}},
    Z,
    p,
    X,
    Y,
    backend::AbstractRiemannianDiffBackend,
) where {𝔽}
    emb_Z = apply_operator!(
        LeviCivitaConnection(decorated_manifold(F.manifold)),
        Z,
        embed(F.manifold, p),
        embed(F.manifold, p, X),
        q -> embed(F.manifold, q, Y(q)),
        backend,
    )
    return project!(F.manifold, Z, p, emb_Z)
end

"""
    apply_operator(
        F::HessianOperator,
        p,
        X,
        f,
        diff_backend::AbstractRiemannianDiffBackend,
        grad_backend::AbstractRiemannianDiffBackend,
    )

Apply the Hessian operator `F` at point `p` to function `f` in the direction `X` using
given differentiation backends: `diff_backend` is used for connection-related
differentiation and `grad_backend` is used for gradient-related differentiation.
"""
apply_operator(F::HessianOperator, p, X, f)

function apply_operator!(
    F::HessianOperator,
    Z,
    p,
    X,
    f,
    diff_backend::AbstractRiemannianDiffBackend,
    grad_backend::AbstractRiemannianDiffBackend,
)
    return apply_operator!(
        F.conn,
        Z,
        p,
        X,
        q -> gradient(base_manifold(F.conn), f, q, grad_backend),
        diff_backend,
    )
end

function apply_operator(
    F::HessianOperator,
    p,
    X,
    f,
    diff_backend::AbstractRiemannianDiffBackend,
    grad_backend::AbstractRiemannianDiffBackend,
)
    return apply_operator(
        F.conn,
        p,
        X,
        let M = base_manifold(F.conn),
            f = f,
            grad_backend = grad_backend
            
            q -> gradient(M, f, q, grad_backend)
        end,
        diff_backend,
    )
end

base_manifold(F::HessianOperator) = base_manifold(F.conn)

function apply_operator(F::HessianOperator, p, X, f)
    return apply_operator(F, p, X, f, rdifferential_backend(), rgradient_backend())
end

function apply_operator!(F::HessianOperator, Z, p, X, Y)
    return apply_operator!(F, Z, p, X, Y, rdifferential_backend(), rgradient_backend())
end
