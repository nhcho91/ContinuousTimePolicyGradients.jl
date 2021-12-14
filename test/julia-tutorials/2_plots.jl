using Plots
using Transducers: Map  # using Transducers.Map as Map


"""
`Transducers.jl` is a very useful data-manipulation tool.
It deals with "iterator" very effectively.
"""
function main()
    ts = 0:0.01:1
    xs = ts |> Map(t -> 2*t) |> collect  # Transducers.Map
    fig = plot(ts, xs)  # Plots.plot
    savefig("example.png")  # save the current figure
    savefig(fig, "example.pdf")  # or, specify which figure you will save
    display(fig)
end
