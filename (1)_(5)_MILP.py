import networkx as nx
from pyomo.environ import *
import math
import random

def build_schedule_matrix(edges, max_slots=100):
    """
    按照论文中描述构建调度矩阵 A
    使用 Protocol Interference Model：两个链路如果共享端点则冲突
    """
    def is_conflict(link1, link2):
        return bool(set(link1) & set(link2))  # 有交集表示冲突

    edge_to_idx = {e: i for i, e in enumerate(edges)} #字典
    edge_covered = {e: False for e in edges} #字典
    all_slots = [] #列表

    while not all(all_covered := list(edge_covered.values())):
        candidate_edges = list(edge for edge in edges if not edge_covered[edge])
        random.shuffle(candidate_edges) #打乱顺序

        Z = []  # 当前 transmission set
        for l in candidate_edges:
            if all(not is_conflict(l, z) for z in Z):
                Z.append(l)

        # 添加 transmission set
        all_slots.append(Z)
        for e in Z:
            edge_covered[e] = True

        if len(all_slots) > max_slots:
            break

    # 构建矩阵 A（|E| × N_slots）
    # 创建了一个矩阵，每一行元素（一条链路的节点）都是0，每一列元素（可同时传输的节点）都是0
    A = [[0 for _ in range(len(all_slots))] for _ in range(len(edges))]
    for slot_idx, slot in enumerate(all_slots):
        for edge in slot:
            i = edge_to_idx[edge]
            A[i][slot_idx] = 1

    return A, len(all_slots)


# 参数设置和决策变量
N = 30                      # 节点数
S_num = 3                   # 源节点数
AC_limit = 3                # AC 数量限制
C = 250                     # 链路容量：kbs
sigma, rho, tau = 150e-9, 300e-9, 300e-9 # J/bit
beta = 30                   # AC 能量提升 J/s(W)

#使得每次随机生成的数字是一样的，影响接下来的random函数
random.seed(42)

# 生成拓扑
# 使用Watts-Strogatz模型生成小网络
# N: 节点总数（在你的代码中是30）
# 3: 每个节点初始的邻居数
# 0.3: 重连概率（每条边以0.3的概率被重新连接）
# tries=100: 尝试生成连通图的最大次数
G = nx.connected_watts_strogatz_graph(N, 3, 0.3, tries=100)
nodes = list(G.nodes()) #获取节点列表
sink = max(nodes) + 1  # all_nodes = nodes + [sink]
G.add_node(sink) # 添加汇聚节点

sources = random.sample(nodes, S_num)

# 构建有向边
edges = []
for i in nodes: # 遍历所有节点
    for j in G.neighbors(i): # 遍历节点i的所有邻居
        if i != j: #不是自环
            edges.append((i, j))
            edges.append((j, i))

# 添加连接 sink 的边（任意 5 个中继节点连接 sink）
relay_candidates = list(set(nodes) - set(sources))
relay_to_sink = random.sample(relay_candidates, 5)  # 改变这个数值，会影响最终结果
for r in relay_to_sink:
    edges.append((r, sink))

# 邻接方向集
all_nodes = nodes + [sink]
N_plus = {i: [] for i in all_nodes}
N_minus = {i: [] for i in all_nodes}
for (i, j) in edges:
    N_plus[i].append(j)
    N_minus[j].append(i)

# 调度矩阵 A 构建
A_matrix, num_slots = build_schedule_matrix(edges)


# 能量采集：论文模拟正弦日照
E = {}
for idx, i in enumerate(nodes):
    t = idx / (N - 1) * math.pi
    irradiance = 75 * math.sin(t)  # mW
    E[i] = irradiance / 1000 # 转为 J/s(W)


# 建立 Pyomo 模型
# indexes
model = ConcreteModel()
model.N = Set(initialize=nodes)
model.S = Set(initialize=sources)
model.Links = Set(initialize=edges, dimen=2)
model.Slots = RangeSet(0, num_slots - 1)

# 变量
model.g = Var(model.S, domain=(0, C))          # 源节点速率 bps
model.f = Var(model.Links, domain=(0, C))      # 链路流量 bps
model.R = Var(model.N, domain={0,1})           # 是否放置AC
model.r = Var(domain=(0,None))                         # 最小源速率目标
model.x = Var(model.Slots, bounds=(0, 1))      # 时隙激活比例

# 目标函数
model.obj = Objective(expr=model.r, sense=maximize)
model.min_rate = ConstraintList()
for i in model.S:
    model.min_rate.add(model.g[i] >= model.r)

# 能量约束
def energy_rule(m, i):
    recv = sum(m.f[(u, i)] for u in N_minus[i] if (u, i) in m.Links)
    send = sum(m.f[(i, v)] for v in N_plus[i] if (i, v) in m.Links)
    sense = m.g[i] if i in m.S else 0
    return sigma * sense + rho * recv + tau * send <= E[i] + beta * m.R[i]
model.energy = Constraint(model.N, rule=energy_rule)

# 流量守恒
def flow_rule(m, i):
    in_flow = sum(m.f[(u, i)] for u in N_minus[i] if (u, i) in m.Links)
    out_flow = sum(m.f[(i, v)] for v in N_plus[i] if (i, v) in m.Links)
    if i == sink:
        return in_flow + (m.g[i] if i in m.S else 0) == 0
    else:
        return in_flow + (m.g[i] if i in m.S else 0) == out_flow
model.flow = Constraint(model.N, rule=flow_rule)

# 链路容量限制（含调度矩阵）
model.cap = ConstraintList()
edge_to_idx = {e: i for i, e in enumerate(edges)}
for (i, j) in edges:
    l_idx = edge_to_idx[(i, j)]
    sched_expr = sum(A_matrix[l_idx][n] * model.x[n] for n in model.Slots)
    model.cap.add(model.f[i, j] <= C * sched_expr)

# 时隙归一化约束
model.slot_sum = Constraint(expr=sum(model.x[n] for n in model.Slots) == 1)

# AC 数量限制
model.ac_limit = Constraint(expr=sum(model.R[i] for i in nodes) <= AC_limit)


# 求解
solver = SolverFactory('gurobi')
results = solver.solve(model,tee=True)

# 输出结果
print("\n=== 求解状态 ===")
print(f"Status: {results.solver.status}")
print(f"Termination Condition: {results.solver.termination_condition}")

if results.solver.termination_condition == TerminationCondition.optimal:
    print("\nMILP 求解完成")
    print(f"最小源节点速率 r = {value(model.r):.2f} kbs")
    for i in model.S:
        print(f"g[{i}] = {value(model.g[i]):.2f} kbs")
    print("放置 AC 的节点：")
    for i in model.N:
        if value(model.R[i]) > 0.5:
            print(f"- Node {i}")
else:
    print("求解失败：模型不可行或未达最优")
