@doc raw"""
    solve_inverse_retract!(
        M,
        X,
        p,
        q,
        X0 = zero_tangent_vector(M, p),
        retraction_method = ExponentialRetraction(),
    )

Given a manifold `M`, two points `p` and `q` on the manifold, and an initial guess `X0` at
the vector $X ∈ T_p M$, such that `q == retract(M, p, X)`, solve for `X` in-place using a
nonlinear solver. Note that this method is not guaranteed to succeed and probably will not
unless `q` is in some small neighborhood around `p` and the initial guess `X0` is good.
If the solver fails to converge, a warning is raised.
"""
function solve_inverse_retract!(
    M,
    X,
    p,
    q,
    X0 = zero_tangent_vector(M, p),
    retraction_method = ExponentialRetraction(),
)
    function f!(F, X)
        retract!(M, F, p, X, retraction_method)
        F .-= q
        return F
    end
    res = nlsolve(f!, X0)
    res.f_converged ||
        @warn "Inverse retraction did not converge after $(res.iterations) iterations."
    return copyto!(X, res.zero)
end
