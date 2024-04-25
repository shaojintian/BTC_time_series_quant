import numpy as np
def diff_filter(x, s, tol):
                """相似性过滤器，x:因子集，s:因子得分，tol:相似性容忍度，输出mask[需剪除的因子为True]"""
                n = len(s) 
                if n <= 1:
                    return [False] * n
                else:
                    corr = abs(np.corrcoef(x))  # 因子矩阵的相关性计算
                    corr[np.eye(n, dtype=bool)] = 0  # 对角线元素重制为0
                    ixr, ixo, ixs = np.arange(n), [], np.zeros(n, dtype=bool)
                    for i in np.argsort(s)[::-1]:
                        if i not in ixo:  # 如果因子已被剪除，不再剪除与之相似性超限的其他因子
                            ixo.extend(ixr[corr[i] > 1 - tol])
                    ixs[ixo] = True
                    return ixs