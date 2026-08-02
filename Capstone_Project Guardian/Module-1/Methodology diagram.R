library(DiagrammeR)

grViz("

digraph Guardian {

graph [
layout = dot,
rankdir = TB,
splines = ortho,
nodesep = 0.45,
ranksep = 0.65,
bgcolor = white
]

node[
shape = box,
style = 'rounded,filled',
fontname = Cambria,
fontsize = 20,
fontcolor = '#111111',
penwidth = 2,
color = '#444444',
width = 5.2,
height = 0.9,
margin = 0.20
]

A[
label='Disaster Dataset\nCollection',
fillcolor='#9FD5FF'
]

B[
label='Data Preprocessing\n& Feature Engineering',
fillcolor='#A8F0C6'
]

C[
label='DBSCAN\nHotspot Detection',
fillcolor='#FFE38A'
]

D[
label='Hybrid Agent-Based\nSimulation',
fillcolor='#FFA9A9'
]

E[
label='PPO Reinforcement\nLearning',
fillcolor='#D9B8FF'
]

F[
label='Intelligent Resource\nAllocation & Evaluation',
fillcolor='#FFC98C'
]

G[
label='Visualization\n& Final Findings',
fillcolor='#7ED6F8'
]

edge[
arrowsize=0.9,
penwidth=3,
color='#1F5AA6',
fontname=Cambria,
fontsize=16,
fontcolor='#222222'
]

A -> B [label=' Prepare Data ']

B -> C [label=' Detect Risk Zones ']

C -> D [label=' Initialize Simulation ']

D -> E [label=' Learn Optimal Policy ']

E -> F [label=' Allocate Resources ']

F -> G [label=' Analyze Results ']

}
")