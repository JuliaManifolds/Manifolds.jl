function solve_exp_ode(
    M::AbstractConnectionManifold,
    p,
    X;
    basis::AbstractBasis=DefaultOrthonormalBasis(),
    solver=AutoVern9(Rodas5()),
    backend=default_differential_backend(),
    retraction::AbstractRetractionMethod=ManifoldsBase.default_retraction_method(M),
    kwargs...,
)
    d = manifold_dimension(M)
    iv = SVector{d}(1:d)
    ix = SVector{d}((d + 1):(2 * d))
    u0 = allocate(p, 2 * d)
    u0[iv] .= X
    u0[ix] .= p

    function exp_problem(u, params, t)
        M = params[1]
        dx = u[iv]
        q = u[ix]
        ddx = allocate(u, Size(d))
        du = allocate(u)
        Γ = christoffel_symbols_second(M, q, basis; backend=backend, retraction=retraction)
        @einsum ddx[k] = -Γ[k, i, j] * dx[i] * dx[j]
        du[iv] .= ddx
        du[ix] .= dx
        return Base.convert(typeof(u), du)
    end
    params = (M,)
    prob = ODEProblem(exp_problem, u0, (0.0, 1.0), params)
    sol = solve(prob, solver; kwargs...)
    q = sol.u[1][(d + 1):(2 * d)]
    return q
end
