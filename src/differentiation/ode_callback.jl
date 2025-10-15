"""
    IntegratorTerminatorNearChartBoundary{TKwargs}

An object for determining the point at which integration of a differential equation
in a chart on a manifold should be terminated for the purpose of switching a chart.

The value stored in `check_chart_switch_kwargs` will be passed as keyword arguments
to  [`check_chart_switch`](@ref). By default an empty tuple is stored.
"""
struct IntegratorTerminatorNearChartBoundary{TKwargs}
    check_chart_switch_kwargs::TKwargs
end

function IntegratorTerminatorNearChartBoundary()
    return IntegratorTerminatorNearChartBoundary(NamedTuple())
end
