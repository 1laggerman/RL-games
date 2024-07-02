import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt

G = nx.DiGraph()

idx = 0
G.add_node(idx, name="ROOT")

idx += 1

for i in range(5):
    G.add_node(idx, name="Child_%i" % i)
    idx += 1
    G.add_node(idx, name="Grandchild_%i" % i)
    idx += 1
    G.add_node(idx, name="Greatgrandchild_%i" % i)
    idx += 1

    G.add_edge(0,  idx-3)
    G.add_edge(idx-3, idx-2)
    G.add_edge(idx-2, idx-1)

# G.add_node(0)
# G.add_node(1)

# G.add_edge(0, 1)

# write dot file to use with graphviz
# run "dot -Tpng test.dot >test.png"
write_dot(G,'test.dot')

# same layout using matplotlib with no labels
plt.title('draw_networkx')
pos = graphviz_layout(G, prog='dot')
nx.draw(G, pos, with_names=True, arrows=True, node_size=1000)
plt.savefig('nx_test.png')