# Testing

Documentation for testing utilities for `Manifolds.jl`. The function [`test_manifold`](@ref Manifolds.ManifoldTests.test_manifold)
can be used to verify that your manifold correctly implements the `Manifolds.jl`
interface. Similarly [`test_group`](@ref Manifolds.ManifoldTests.test_group) and [`test_action`](@ref Manifolds.ManifoldTests.test_action) can be used to verify implementation of groups and group actions.

```@autodocs
Modules = [Manifolds, Manifolds.ManifoldTests]
Pages = ["tests/ManifoldTests.jl", "tests/tests_general.jl", "tests/tests_group.jl"]
Order = [:type, :function]
```
