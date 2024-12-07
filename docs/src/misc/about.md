# About `Manifolds.jl`

`Manifolds.jl` was started by [Seth Axen](https://github.com/sethaxen), [Mateusz Baran](https://github.com/mateuszbaran), [Ronny Bergmann](https://github.com/kellertuer), and [Antoine Levitt](https://github.com/antoine-levitt) in 2019, after a very fruitful discussion following the release of the first version of [`Manopt.jl`](https://manoptjl.org/). The goal of `Manifolds.jl` is to provide a library of manifolds in Julia. The manifolds are implemented using the [´ManifoldsBase.jl](https:// github.com/JuliaManifolds/ManifoldsBase.jl/) interface.

## Main developers

- [Mateusz Baran](https://github.com/mateuszbaran)
- [Ronny Bergmann](https://github.com/kellertuer)

## Former Main Developers

- [Seth Axen](https://github.com/sethaxen)

## Contributors
(in alphabetical order)

- [Nick Dewaele](https://github.com/Nikdwal) contributed the [Tucker manifold](manifolds/tucker.md)
- [Renée Dornig](https://github.com/r-dornig) contributed the [centered  matrices](manifolds/centered.md) and the [essential manifold](manidfold/essential.md)
- [David Hong](https://github.com/dahong67) contributed uniform distributions on the Stiefel and Grassmann manifolds.
- [Johannes Voll Kolstø]() contributed the [symplectic manifold](manifolds/symplectic.md), the [symplectic Stiefel manifold](manifolds/symplecticstiefel.md)
- [Manuel Weiß](https://github.com/manuelweisser) contributed [symmetric matrices](manifolds/symmetric.md)

as well as everyone else reporting, investigating, and fixing bugs or fixing typographical errors in the documentation, see the [GitHub contributors page](https://github.com/JuliaManifolds/Manifolds.jl/graphs/contributors).

Of course all further [contributions](CONTRIBUTING.md) are always welcome!

## Projects using `Manifolds.jl`

- [Caesar.jl](https://juliarobotics.org/Caesar.jl/latest/concepts/using_manifolds/)
- [ManoptExamples.jl](https://github.com/JuliaManifolds/ManoptExamples.jl) collecting examples of optimization problems on manifolds implemented using `Manifolds.jl` and [`Manopt.jl`](https://manoptjl.org).

Do you use Manifolds.jl in you package? Let us know and open an [issue](https://github.com/JuliaManifolds/Manifolds.jl/issues/new/choose) or [pull request](https://github.com/JuliaManifolds/Manifolds.jl/compare) to add it to the list!

## License

[MIT License](https://github.com/JuliaManifolds/Manifolds.jl/blob/master/LICENSE)