function solve_exp_ode(
    M::AbstractManifold,
    p,
    X;
    basis::AbstractBasis=DefaultOrthonormalBasis(),
    solver=AutoVern9(Rodas5()),
    backend=default_differential_backend(),
    retraction::AbstractRetractionMethod=ManifoldsBase.default_retraction_method(M),
    kwargs...,
)
    n = length(p)
    iv = SVector{n}(1:n)
    ix = SVector{n}((n + 1):(2 * n))
    u0 = allocate(p, 2 * n)
    u0[iv] .= get_get_coordinates!(M, u0[iv])
    u0[ix] .= p

    function exp_problem(u, params, t)
        M = params[1]
        q = u[ix]
        dx = u[iv]
        ddx = allocate(u, Size(n))
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
    q = sol.u[1][(n + 1):(2 * n)]
    return q
end
