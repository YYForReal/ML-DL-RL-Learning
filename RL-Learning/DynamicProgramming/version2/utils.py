def print_agent(agent, end=[], action_meaning=['^', 'v', '<', '>']):
    print_v(agent.v, agent.env.nrow, agent.env.ncol)
    print_pi(agent.pi, agent.env.nrow, agent.env.ncol, end, action_meaning)


def print_v(v, nrow, ncol):
    print("状态价值：")
    for i in range(nrow):
        for j in range(ncol):
            # 为了输出美观,保持输出6个字符
            print('%6.6s' %
                  ('%.3f' % v[i * ncol + j]), end=' ')
        print()


def print_pi(pi, nrow, ncol, end=[], action_meaning=['^', 'v', '<', '>']):
    print("策略：")
    for i in range(nrow):
        for j in range(ncol):
            # 目标状态输出oooo
            if (i * ncol + j) in end:  #
                print('oooo', end=' ')
            else:
                a = pi[i * ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()
