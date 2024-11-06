#!/usr/bin/env julia
#
#

if "--help" ∈ ARGS
    println(
        """
docs/make.jl

Render the `Manopt.jl` documentation with optional arguments

Arguments
* `--exclude-tutorials` - exclude the tutorials from the menu of Documenter,
  this can be used if you do not have Quarto installed to still be able to render the docs
  locally on this machine. This option should not be set on CI.
* `--help`              - print this help and exit without rendering the documentation
* `--prettyurls`        – toggle the prettyurls part to true (which is otherwise only true on CI)
* `--quarto`            – run the Quarto notebooks from the `tutorials/` folder before generating the documentation
  this has to be run locally at least once for the `tutorials/*.md` files to exist that are included in
  the documentation (see `--exclude-tutorials`) for the alternative.
  If they are generated once they are cached accordingly.
  Then you can spare time in the rendering by not passing this argument.
  If quarto is not run, some tutorials are generated as empty files, since they
  are referenced from within the documentation. These are currently
  `Optimize.md` and `ImplementOwnManifold.md`.
""",
    )
    exit(0)
end

#
# (a) if docs is not the current active environment, switch to it
# (from https://github.com/JuliaIO/HDF5.jl/pull/1020/) 
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(; path=(@__DIR__) * "/../"))
    Pkg.resolve()
    Pkg.instantiate()
end

# (b) Did someone say render?
if "--quarto" ∈ ARGS
    using CondaPkg
    CondaPkg.withenv() do
        @info "Rendering Quarto"
        tutorials_folder = (@__DIR__) * "/../tutorials"
        # instantiate the tutorials environment if necessary
        Pkg.activate(tutorials_folder)
        # For a breaking release -> also set the tutorials folder to the most recent version
        Pkg.develop(PackageSpec(; path=(@__DIR__) * "/../"))
        Pkg.resolve()
        Pkg.instantiate()
        Pkg.build("IJulia") # build `IJulia` to the right version.
        Pkg.activate(@__DIR__) # but return to the docs one before
        run(`quarto render $(tutorials_folder)`)
        return nothing
    end
else # fallback to at least create empty files for the start tutorial since that is linked
    touch(joinpath(@__DIR__, "src/tutorials/getstarted.md"))
end

tutorials_in_menu = true
if "--exclude-tutorials" ∈ ARGS
    @warn """
    You are excluding the tutorials from the Menu,
    which might be done if you can not render them locally.

    Remember that this should never be done on CI for the full documentation.
    """
    tutorials_in_menu = false
end

# (c) load necessary packages for the docs
using Plots, RecipesBase, Manifolds, ManifoldsBase, Documenter, PythonPlot
using DocumenterCitations, DocumenterInterLinks
# required for loading methods that handle differential equation solving
using OrdinaryDiffEq, BoundaryValueDiffEq, DiffEqCallbacks
using NLsolve
# required for loading the manifold tests functions
using Test, FiniteDifferences
ENV["GKSwstype"] = "100"

# (d) add CONTRIBUTING.md and NEWS.md to docs
generated_path = joinpath(@__DIR__, "src", "misc")
base_url = "https://github.com/JuliaManifolds/Manifolds.jl/blob/master/"
isdir(generated_path) || mkdir(generated_path)
for fname in ["CONTRIBUTING.md", "NEWS.md"]
    open(joinpath(generated_path, fname), "w") do io
        # Point to source license file
        println(
            io,
            """
            ```@meta
            EditURL = "$(base_url)$(fname)"
            ```
            """,
        )
        # Write the contents out below the meta block
        for line in eachline(joinpath(dirname(@__DIR__), fname))
            println(io, line)
        end
    end
end

# (e) build the tutorials menu
tutorials_menu =
    "How to..." => [
        "🚀 Get Started with `Manifolds.jl`" => "tutorials/getstarted.md",
        "work in charts" => "tutorials/working-in-charts.md",
        "perform Hand gesture analysis" => "tutorials/hand-gestures.md",
        "integrate on manifolds and handle probability densities" => "tutorials/integration.md",
        "explore curvature without coordinates" => "tutorials/exploring-curvature.md",
        "work with groups" => "tutorials/groups.md",
    ]
# (f) final step: render the docs

bib = CitationBibliography(joinpath(@__DIR__, "src", "references.bib"); style=:alpha)
links = InterLinks(
    "ManifoldsBase" => ("https://juliamanifolds.github.io/ManifoldsBase.jl/stable/"),
)
modules = [
    Manifolds,
    isdefined(Base, :get_extension) ?
    Base.get_extension(Manifolds, :ManifoldsBoundaryValueDiffEqExt) :
    Manifolds.ManifoldsBoundaryValueDiffEqExt,
    isdefined(Base, :get_extension) ?
    Base.get_extension(Manifolds, :ManifoldsNLsolveExt) : Manifolds.ManifoldsNLsolveExt,
    isdefined(Base, :get_extension) ?
    Base.get_extension(Manifolds, :ManifoldsOrdinaryDiffEqDiffEqCallbacksExt) :
    Manifolds.ManifoldsOrdinaryDiffEqDiffEqCallbacksExt,
    isdefined(Base, :get_extension) ?
    Base.get_extension(Manifolds, :ManifoldsOrdinaryDiffEqExt) :
    Manifolds.ManifoldsOrdinaryDiffEqExt,
    isdefined(Base, :get_extension) ?
    Base.get_extension(Manifolds, :ManifoldsRecipesBaseExt) :
    Manifolds.ManifoldsRecipesBaseExt,
    isdefined(Base, :get_extension) ? Base.get_extension(Manifolds, :ManifoldsTestExt) :
    Manifolds.ManifoldsTestExt,
]
if modules isa Vector{Union{Nothing,Module}}
    error("At least one module has not been properly loaded: ", modules)
end
makedocs(;
    format=Documenter.HTML(
        prettyurls=(get(ENV, "CI", nothing) == "true") || ("--prettyurls" ∈ ARGS),
        assets=["assets/favicon.ico", "assets/citations.css"],
        size_threshold_warn=200 * 2^10, # raise slightly from 100 to 200 KiB
        size_threshold=300 * 2^10,      # raise slightly 200 to 300 KiB
    ),
    modules=modules,
    authors="Seth Axen, Mateusz Baran, Ronny Bergmann, and contributors.",
    sitename="Manifolds.jl",
    pages=[
        "Home" => "index.md",
        (tutorials_in_menu ? [tutorials_menu] : [])...,
        "Manifolds" => [
            "Basic manifolds" => [
                "Centered matrices" => "manifolds/centeredmatrices.md",
                "Cholesky space" => "manifolds/choleskyspace.md",
                "Circle" => "manifolds/circle.md",
                "Elliptope" => "manifolds/elliptope.md",
                "Essential manifold" => "manifolds/essentialmanifold.md",
                "Euclidean" => "manifolds/euclidean.md",
                "Fixed-rank matrices" => "manifolds/fixedrankmatrices.md",
                "Flag" => "manifolds/flag.md",
                "Generalized Stiefel" => "manifolds/generalizedstiefel.md",
                "Generalized Grassmann" => "manifolds/generalizedgrassmann.md",
                "Grassmann" => "manifolds/grassmann.md",
                "Hamiltonian" => "manifolds/hamiltonian.md",
                "Hyperbolic space" => "manifolds/hyperbolic.md",
                "Hyperrectangle" => "manifolds/hyperrectangle.md",
                "Invertible matrices" => "manifolds/invertible.md",
                "Lorentzian manifold" => "manifolds/lorentz.md",
                "Multinomial doubly stochastic matrices" => "manifolds/multinomialdoublystochastic.md",
                "Multinomial matrices" => "manifolds/multinomial.md",
                "Multinomial symmetric matrices" => "manifolds/multinomialsymmetric.md",
                "Multinomial symmetric positive definite matrices" => "manifolds/multinomialsymmetricpositivedefinite.md",
                "Oblique manifold" => "manifolds/oblique.md",
                "Probability simplex" => "manifolds/probabilitysimplex.md",
                "Positive numbers" => "manifolds/positivenumbers.md",
                "Projective space" => "manifolds/projectivespace.md",
                "Orthogonal and Unitary Matrices" => "manifolds/generalunitary.md",
                "Rotations" => "manifolds/rotations.md",
                "Shape spaces" => "manifolds/shapespace.md",
                "Skew-Hermitian matrices" => "manifolds/skewhermitian.md",
                "Spectrahedron" => "manifolds/spectrahedron.md",
                "Sphere" => "manifolds/sphere.md",
                "Stiefel" => "manifolds/stiefel.md",
                "Symmetric matrices" => "manifolds/symmetric.md",
                "Symmetric positive definite" => "manifolds/symmetricpositivedefinite.md",
                "SPD, fixed determinant" => "manifolds/spdfixeddeterminant.md",
                "Symmetric positive semidefinite fixed rank" => "manifolds/symmetricpsdfixedrank.md",
                "Symplectic Grassmann" => "manifolds/symplecticgrassmann.md",
                "Symplectic matrices" => "manifolds/symplectic.md",
                "Symplectic Stiefel" => "manifolds/symplecticstiefel.md",
                "Torus" => "manifolds/torus.md",
                "Tucker" => "manifolds/tucker.md",
                "Unit-norm symmetric matrices" => "manifolds/spheresymmetricmatrices.md",
            ],
            "Combined manifolds" => [
                "Fiber bundle" => "manifolds/fiber_bundle.md",
                "Graph manifold" => "manifolds/graph.md",
                "Power manifold" => "manifolds/power.md",
                "Product manifold" => "manifolds/product.md",
                "Vector bundle" => "manifolds/vector_bundle.md",
            ],
            "Manifold decorators" => [
                "Connection manifold" => "manifolds/connection.md",
                "Group manifold" => "manifolds/group.md",
                "Metric manifold" => "manifolds/metric.md",
                "Quotient manifold" => "manifolds/quotient.md",
            ],
        ],
        "Features on Manifolds" => [
            "Atlases and charts" => "features/atlases.md",
            "Differentiation" => "features/differentiation.md",
            "Distributions" => "features/distributions.md",
            "Group actions" => "features/group_actions.md",
            "Integration" => "features/integration.md",
            "Statistics" => "features/statistics.md",
            "Testing" => "features/testing.md",
            "Utilities" => "features/utilities.md",
        ],
        "Miscellanea" => [
            "About" => "misc/about.md",
            "Changelog" => "misc/NEWS.md",
            "Contributing" => "misc/CONTRIBUTING.md",
            "Internals" => "misc/internals.md",
            "Notation" => "misc/notation.md",
            "References" => "misc/references.md",
        ],
    ],
    plugins=[bib, links],
    warnonly=[:missing_docs],
)
deploydocs(repo="github.com/JuliaManifolds/Manifolds.jl.git", push_preview=true)
