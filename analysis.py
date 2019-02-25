import matplotlib.pyplot as plt


# line_types = ['b:', 'r--', 'y-.', 'g-']
# algorithms = ['RLN', 'RLNL', 'RIM', 'RLV']
line_types = ['b:']
algorithms = ['RLN']
metrics = {'acceptance ratio': 'Acceptance Ratio',
           'average revenue': 'Long Time Average Revenue',
           'average cost': 'Long Time Average Cost',
           'R_C': 'Long Time Revenue/Cost Ratio',
           'average node utilization': 'Average Node Utilization',
           'average link utilization': 'Average Link Utilization'}




def save_result(filename,evaluations):
    """将一段时间内底层网络的性能指标输出到指定文件内"""

    filename = 'Results/%s.txt' % filename
    with open(filename, 'w') as f:
        for time, evaluation in evaluations.items():
            f.write("%-10s\t" % time)
            f.write("%-20s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s\n" % evaluation)


def read_result(filename):
    """读取结果文件"""

    filename = 'Results/%s.txt' % filename
    with open(filename) as f:
        lines = f.readlines()

    t, acceptance, revenue, cost, r_to_c, nodeuti, linkuti = [], [], [], [], [], [], []
    for line in lines:
        a, b, c, d, e, f, g = [float(x) for x in line.split()]
        t.append(a)
        acceptance.append(b)
        revenue.append(c / a)
        cost.append(d / a)
        r_to_c.append(e)
        nodeuti.append(f)
        linkuti.append(g)

    return t, acceptance, revenue, cost, r_to_c, nodeuti, linkuti


def draw():
    """绘制实验结果图"""

    results = []
    for alg in algorithms:
        results.append(read_result(alg))

    index = 0
    for metric, title in metrics.items():
        index += 1
        plt.figure()
        for alg_id in range(len(algorithms)):
            x = results[alg_id][0]
            y = results[alg_id][index]
            plt.plot(x, y, line_types[alg_id], label=algorithms[alg_id])
        plt.xlim([0, 55000])
        plt.xlabel("time", fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.title(title, fontsize=15)
        plt.legend(loc='best', fontsize=12)
        plt.savefig('Results/%s.jpg' %metric)

def lossanaly():
    pass


if __name__ == '__main__':
    draw()