function solve_exp_ode(
    M::AbstractManifold,
    p,
    X;
    basis::AbstractBasis=DefaultOrthonormalBasis(),
    solver=AutoVern9(Rodas5()),
    backend=default_differential_backend(),
    kwargs...,
)
    n = length(p)
    iv = SVector{n}(1:n)
    ix = SVector{n}((n + 1):(2n))
    u0 = allocate(p, 2n)
    u0[iv] .= X
    u0[ix] .= p

    function exp_problem(u, params, t)
        M = params[1]
        dx = u[iv]
        p = u[ix]
        ddx = allocate(u, Size(n))
        du = allocate(u)
        Γ = christoffel_symbols_second(M, p, basis; backend=backend)
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
# also define exp! for metric manifold anew in this case
function exp!(
    ::TraitList{IsMetricManifold},
    M::AbstractDecoratorManifold,
    q,
    p,
    X;
    kwargs...,
)
    copyto!(M, q, solve_exp_ode(M, p, X; kwargs...))
    return q
end
