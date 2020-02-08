function solve_exp_ode(
    M::MetricManifold,
    x,
    v,
    tspan;
    solver = AutoVern9(Rodas5()),
    backend = diff_backend(),
    kwargs...,
)
    n = length(x)
    iv = SVector{n}(1:n)
    ix = SVector{n}(n+1:2n)
    u0 = allocate(x, 2n)
    u0[iv] .= v
    u0[ix] .= x

    function exp_problem(u, p, t)
        M = p[1]
        dx = u[iv]
        x = u[ix]
        ddx = allocate(u, Size(n))
        du = allocate(u)
        Γ = christoffel_symbols_second(M, x; backend = backend)
        @einsum ddx[k] = -Γ[k, i, j] * dx[i] * dx[j]
        du[iv] .= ddx
        du[ix] .= dx
        return Base.convert(typeof(u), du)
    end

    p = (M,)
    prob = ODEProblem(exp_problem, u0, tspan, p)
    sol = solve(prob, solver; kwargs...)
    return sol
end
