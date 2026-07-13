library(DiagrammeR)

grViz("
digraph MDP {

graph [
layout = dot
rankdir = TB
splines = ortho
nodesep = 0.5
ranksep = 0.8
bgcolor = white
]

node [
shape = rectangle
style = 'rounded,filled'
fontname = Helvetica
fontsize = 15
fontcolor = black
color = black
penwidth = 2
margin = 0.18
]

Start [
label = 'Markov Decision Process'
fillcolor = '#1F77B4'
fontcolor = white
fontsize = 20
]

Problem [
label = 'Define Problem & Objective'
fillcolor = '#AED6F1'
]

Components [
label = 'MDP Components\\nStates, Actions, Transition, Reward, Discount'
fillcolor = '#A9DFBF'
]

Process [
label = 'Decision Cycle\\nObserve → Act → Reward → Next State'
fillcolor = '#F9E79F'
]

Update [
label = 'Update Value / Policy'
fillcolor = '#FAD7A0'
]

Converge [
label = 'Converged?'
shape = diamond
fillcolor = '#F8C471'
]

Policy [
label = 'Optimal Policy & Value'
fillcolor = '#F1948A'
]

Decision [
label = 'Best Sequential Decision'
fillcolor = '#EC7063'
fontcolor = white
]

Start -> Problem
Problem -> Components
Components -> Process
Process -> Update
Update -> Converge
Converge -> Process [label='No']
Converge -> Policy [label='Yes']
Policy -> Decision

}
")
