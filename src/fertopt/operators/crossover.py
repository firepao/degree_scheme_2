from __future__ import annotations

import numpy as np


def build_elite_prototypes(
    population: np.ndarray,
    objective_values: np.ndarray,
    prototype_count: int,
    elite_ratio: float,
    kmeans_iters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    try:
        from deap import base, creator, tools
    except ImportError:
        pass

    if population.ndim != 2:
        raise ValueError("population 必须为二维")
    if objective_values.ndim != 2:
        raise ValueError("objective_values 必须为二维")
    if population.shape[0] != objective_values.shape[0]:
        raise ValueError("population 与 objective_values 样本数必须一致")

    sample_count = population.shape[0]
    elite_num = max(2, int(sample_count * elite_ratio))
    elite_num = min(elite_num, sample_count)

    try:
        from deap import tools, base, creator
        
        # We need a temporary individual class that has fitness attribute
        # We can reuse deap's structure if available, or mock it
        
        # Check if FitnessMin already exists in creator (it should if runner ran)
        # But here we are in a module.
        # Safe way: Create local dummy classes for selection purpose
        
        class LocalFitness(base.Fitness):
            weights = (-1.0,) * objective_values.shape[1]

        class LocalInd(list):
            def __init__(self, values, index):
                self.extend(values)
                self.index = index
                self.fitness = LocalFitness()
                self.fitness.values = tuple(objective_values[index])

        # Create population of LocalInd
        inds = []
        for i in range(sample_count):
            inds.append(LocalInd(population[i], i))

        # Use NSGA-II selection
        # This will select based on rank and crowding distance
        selected = tools.selNSGA2(inds, elite_num)
        elite_indices = [ind.index for ind in selected]
    
    except ImportError:
        # Fallback: Simple Sum of Normalized Objectives
        # Normalize to [0, 1] range first to avoid dominance by large value objectives
        min_vals = np.min(objective_values, axis=0)
        max_vals = np.max(objective_values, axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0 # Avoid div by zero
        
        norm_obj = (objective_values - min_vals) / ranges
        score = np.sum(norm_obj, axis=1)
        elite_indices = np.argsort(score)[:elite_num]

    elites = population[elite_indices]

    k = max(1, min(int(prototype_count), elites.shape[0]))
    return _kmeans_numpy(elites, k=k, iters=max(1, int(kmeans_iters)), rng=rng)


def synergistic_balance_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    stage_count: int,
    alpha: float,  # 控制平衡（总量）交叉的混合程度，类似于平滑系数
    lower_bound: float,
    upper_bound: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    协同平衡交叉 (Synergistic Balance Crossover, SBC)
    
    核心理念：
    中药材（如丹参）的有效成分积累往往依赖于 N、P、K 的特定**比例**。
    此算子将施肥方案解耦为 **“强度（总量）”** 和 **“结构（比例）”** 两个正交的特征进行独立进化。
    
    1. 强度 (Intensity) M = ||X|| (养分总量，决定生物量)。
    2. 结构 (Structure) D = X / M (养分比例向量，决定品质/有效成分)。
    
    M 进行算术/SBX 交叉，D 进行球面线性插值 (Slerp)。
    
    参数:
        parent_a, parent_b: 父代决策变量向量
        stage_count: 施肥阶段数 (假设每个阶段有 N, P, K 3个变量)
        alpha: 强度交叉的混合系数 (0~1)，0.5 为平均，倾向于产生中间总量
        lower_bound, upper_bound: 变量边界
        rng: 随机数生成器
    """
    dim = parent_a.shape[0]
    if dim != parent_b.shape[0]:
        raise ValueError("父代维度必须一致")
    
    vars_per_stage = 3 # N, P, K
    
    child1 = np.copy(parent_a)
    child2 = np.copy(parent_b)
    
    # 遍历每个阶段
    for s in range(stage_count):
        start_idx = s * vars_per_stage
        end_idx = start_idx + vars_per_stage
        
        # 提取当前阶段的 NPK 向量
        vec_a = parent_a[start_idx:end_idx]
        vec_b = parent_b[start_idx:end_idx]
        
        # 1. 计算强度 (Magnitude)
        mag_a = np.linalg.norm(vec_a) + 1e-9 # 避免除零
        mag_b = np.linalg.norm(vec_b) + 1e-9
        
        # 2. 计算方向 (Direction / Structure)
        dir_a = vec_a / mag_a
        dir_b = vec_b / mag_b
        
        # 3. 交叉强度 (Magnitude Crossover)
        # 这里使用高斯扰动的混合，允许探索更大或更小的总量
        beta = rng.uniform(0.0, 1.0) # 强度混合比例
        
        mag_c1 = beta * mag_a + (1 - beta) * mag_b
        mag_c2 = (1 - beta) * mag_a + beta * mag_b
        
        # 4. 交叉方向 (Direction Crossover) - Slerp (Spherical Linear Interpolation)
        dot_val = np.clip(np.dot(dir_a, dir_b), -1.0, 1.0)
        theta = np.arccos(dot_val)
        
        if theta < 1e-6:
            dir_c1 = dir_a
            dir_c2 = dir_b
        else:
            # 引入外推能力 (Extrapolation)
            # t 取值范围从 [0, 1] 扩展到 [-0.2, 1.2]，允许向两侧延伸探索新的配比
            # 这种“过度插值”有助于跳出局部最优配比
            t1 = rng.uniform(-0.2, 1.2)
            t2 = rng.uniform(-0.2, 1.2)
            
            # 由于可能超出 [0, 1]，sin 计算需要注意符号，但 sin(kt) / sin(t) 公式本身对于 k<0 或 k>1 也是数学成立的
            # 只要确保归一化即可。
            
            sin_theta = np.sin(theta)
            
            # 防止 sin_theta 过小导致除零
            if abs(sin_theta) < 1e-9:
                 dir_c1 = dir_a
                 dir_c2 = dir_b
            else:
                w1_a = np.sin((1 - t1) * theta) / sin_theta
                w1_b = np.sin(t1 * theta) / sin_theta
                dir_c1 = w1_a * dir_a + w1_b * dir_b
                
                w2_a = np.sin((1 - t2) * theta) / sin_theta
                w2_b = np.sin(t2 * theta) / sin_theta
                dir_c2 = w2_a * dir_a + w2_b * dir_b
            
            # 归一化确保在球面上
            dir_c1 = dir_c1 / (np.linalg.norm(dir_c1) + 1e-9)
            dir_c2 = dir_c2 / (np.linalg.norm(dir_c2) + 1e-9)
            
        # 5. 重组
        vec_c1 = mag_c1 * dir_c1
        vec_c2 = mag_c2 * dir_c2
        
        child1[start_idx:end_idx] = vec_c1
        child2[start_idx:end_idx] = vec_c2

    # 处理剩余变量 (如果有)
    expected_dim = stage_count * vars_per_stage
    if dim > expected_dim:
        for i in range(expected_dim, dim):
             u = rng.random()
             child1[i] = u * parent_a[i] + (1 - u) * parent_b[i]
             child2[i] = (1 - u) * parent_a[i] + u * parent_b[i]

    # 边界约束
    child1 = np.clip(child1, lower_bound, upper_bound)
    child2 = np.clip(child2, lower_bound, upper_bound)
    
    return child1, child2


def _kmeans_numpy(data: np.ndarray, k: int, iters: int, rng: np.random.Generator) -> np.ndarray:
    if data.shape[0] <= k:
        return data.copy()

    init_idx = rng.choice(data.shape[0], size=k, replace=False)
    centers = data[init_idx].copy()

    for _ in range(iters):
        dist = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dist, axis=1)
        new_centers = centers.copy()

        for idx in range(k):
            cluster_points = data[labels == idx]
            if cluster_points.shape[0] == 0:
                new_centers[idx] = data[rng.integers(0, data.shape[0])]
            else:
                new_centers[idx] = np.mean(cluster_points, axis=0)

        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    return centers


# =============================================================================
# 新增高效交叉算子
# =============================================================================

def sbx_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    eta: float = 20.0,
    lower_bound: float = 0.0,
    upper_bound: float = 300.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    模拟二进制交叉 (Simulated Binary Crossover, SBX)
    
    简介:
        SBX 是 NSGA-II 中的标准交叉算子，模拟二进制单点交叉的行为。
        通过分布指数 eta 控制子代与父代的接近程度。
        eta 值越小，生成的子代距离父代越远（探索性更强）；
        eta 值越大，子代越接近父代（开发性更强）。
    
    数学原理:
        根据父代 x1, x2，生成子代 y1, y2:
        1. 随机生成 u ~ Uniform(0,1)
        2. 如果 u <= 0.5:
           beta = (2*u)^(1/(eta+1))
        3. 否则:
           beta = (1/(2-2*u))^(1/(eta+1))
        4. y1 = 0.5 * ((1+beta)*x1 + (1-beta)*x2)
           y2 = 0.5 * ((1-beta)*x1 + (1+beta)*x2)
    
    优点:
        - 计算效率高：仅使用简单的数学运算
        - 参数少：仅需 eta
        - 理论完善：有严格的数学基础
        - 保持多样性：通过 eta 控制搜索范围
    
    参数:
        parent_a: 第一个父代决策向量
        parent_b: 第二个父代决策向量
        eta: 分布指数，默认20。值越小探索性越强，值越大开发性越强
        lower_bound: 变量下界
        upper_bound: 变量上界
        rng: 随机数生成器，如果为None则使用np.random
    
    返回:
        (child1, child2): 两个子代决策向量
    
    示例:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> parent_a = np.array([100, 50, 80, 120, 60, 90])
        >>> parent_b = np.array([80, 60, 100, 100, 50, 70])
        >>> child1, child2 = sbx_crossover(parent_a, parent_b, eta=20, 
        ...                                lower_bound=0, upper_bound=300, rng=rng)
    """
    # 参数校验
    if parent_a.shape != parent_b.shape:
        raise ValueError("父代维度必须一致")
    if lower_bound >= upper_bound:
        raise ValueError("lower_bound 必须小于 upper_bound")
    
    # 使用随机数生成器
    if rng is None:
        rng = np.random.default_rng()
    
    dim = parent_a.shape[0]
    
    # 初始化子代
    child1 = np.copy(parent_a)
    child2 = np.copy(parent_b)
    
    # 遍历每个维度进行交叉
    for i in range(dim):
        # 随机生成 [0, 1) 之间的值
        u = rng.random()
        
        if u <= 0.5:
            # 使用第一种情况
            beta = (2 * u) ** (1.0 / (eta + 1))
        else:
            # 使用第二种情况
            beta = (1.0 / (2.0 - 2.0 * u)) ** (1.0 / (eta + 1))
        
        # 计算子代
        child1[i] = 0.5 * ((1 + beta) * parent_a[i] + (1 - beta) * parent_b[i])
        child2[i] = 0.5 * ((1 - beta) * parent_a[i] + (1 + beta) * parent_b[i])
    
    # 应用边界约束
    child1 = np.clip(child1, lower_bound, upper_bound)
    child2 = np.clip(child2, lower_bound, upper_bound)
    
    return child1, child2


def adaptive_sbx_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    current_gen: int,
    max_gen: int,
    eta_start: float = 2.0,
    eta_end: float = 30.0,
    lower_bound: float = 0.0,
    upper_bound: float = 300.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    自适应模拟二进制交叉 (Adaptive SBX)
    
    简介:
        随着进化代数增加，线性增加分布指数 eta。
        前期 eta 小 -> 探索性强
        后期 eta 大 -> 开发性强
    """
    if max_gen <= 0:
        progress = 1.0
    else:
        progress = np.clip(current_gen / max_gen, 0.0, 1.0)
        
    eta = eta_start + (eta_end - eta_start) * progress
    
    return sbx_crossover(
        parent_a, parent_b,
        eta=eta,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        rng=rng
    )


def de_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    parent_c: np.ndarray,
    cr: float = 0.9,
    f: float = 0.5,
    lower_bound: float = 0.0,
    upper_bound: float = 300.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    差分进化交叉 (Differential Evolution Crossover, DE/current-to-best/1)
    
    简介:
        DE 交叉源自差分进化算法，以其强大的全局搜索能力著称。
        通过父代个体间的差分向量来生成候选解。
        current-to-best/1 策略结合了当前个体的信息和最优个体的方向。
    
    数学原理:
        已知三个父代 a, b, c，以及可选的最优参考 best:
        1. 差分向量: d = b - c (基础差分)
        2. 扰动: v = a + F * d (如果不使用best)
           或 v = a + F * (best - a) + F1 * (b - c) (current-to-best)
        3. 交叉: 每个维度以概率 cr 从 v 中继承，否则从父代 a 继承
    
    优点:
        - 强全局搜索能力：能有效跳出局部最优
        - 自适应：根据父代差异自动调整搜索方向
        - 高效：计算简单，适合大规模问题
        - 多目标表现好：在CEC测试集上表现优异
    
    参数:
        parent_a: 主父代（目标个体）
        parent_b: 父代B，用于生成差分向量
        parent_c: 父代C，用于生成差分向量
        cr: 交叉概率，默认0.9。值越大继承突变向量的概率越高
        f: 缩放因子，默认0.5。控制差分向量的缩放程度
        lower_bound: 变量下界
        upper_bound: 变量上界
        rng: 随机数生成器
    
    返回:
        (child1, child2): 两个子代（当前实现中child2与child1相同，
                          若需要两个独立子代需调用两次）
    
    示例:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> a = np.array([100, 50, 80, 120, 60, 90])
        >>> b = np.array([80, 60, 100, 100, 50, 70])
        >>> c = np.array([90, 55, 85, 110, 55, 75])
        >>> child1, _ = de_crossover(a, b, c, cr=0.9, f=0.5, 
        ...                          lower_bound=0, upper_bound=300, rng=rng)
    """
    # 参数校验
    if parent_a.shape != parent_b.shape or parent_a.shape != parent_c.shape:
        raise ValueError("所有父代维度必须一致")
    if not (0 <= cr <= 1):
        raise ValueError("交叉概率 cr 必须在 [0, 1] 范围内")
    if not (0 < f <= 1):
        raise ValueError("缩放因子 f 必须在 (0, 1] 范围内")
    
    if rng is None:
        rng = np.random.default_rng()
    
    dim = parent_a.shape[0]
    
    # 生成突变向量: v = a + F * (b - c)
    # 这里使用经典的 DE/rand/1 策略
    diff_vector = f * (parent_b - parent_c)
    mutant = parent_a + diff_vector
    
    # 边界处理: 将超出边界的值进行反弹处理
    # 这种处理方式比简单的裁剪更能保持多样性
    mutant = np.clip(mutant, lower_bound, upper_bound)
    
    # 创建掩码：决定从突变向量还是父代继承
    # 使用维度级别的交叉，而不是个体级别
    cross_mask = rng.random(dim) < cr
    
    # 确保至少有一个维度被修改（DE 特性）
    if not np.any(cross_mask):
        idx = rng.integers(0, dim)
        cross_mask[idx] = True
    
    # 生成子代
    child = np.where(cross_mask, mutant, parent_a)
    
    # 再次边界约束
    child = np.clip(child, lower_bound, upper_bound)
    
    return child, child


def pcx_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    parent_c: np.ndarray | None = None,
    eta: float = 0.5,
    zeta: float = 0.5,
    lower_bound: float = 0.0,
    upper_bound: float = 300.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    父代中心交叉 (Parent-Centric Crossover, PCX)
    
    简介:
        PCX 是一种以父代为中心的交叉算子，由 Bäck 等人提出。
        它保持父代个体的"个性"，同时在父代附近进行搜索。
        特别适合多阶段决策问题，因为它自然保持每个阶段的特征。
    
    数学原理:
        已知两个或三个父代 a, b, c:
        1. 计算父代的中心点: center = (a + b + c) / n
        2. 计算从中心点到父代的偏移向量
        3. 子代在父代附近生成，使用高斯分布扰动
        4. 子代 = 父代 + zeta * (父代 - center) + eta * 随机向量
    
    优点:
        - 保持父代优良特性：不破坏优秀解的结构
        - 适合多阶段问题：自然保持每个阶段的NPK比例
        - 参数稳健：对参数不敏感
        - 局部搜索能力强：在最优解附近表现好
    
    参数:
        parent_a: 第一个父代
        parent_b: 第二个父代
        parent_c: 可选的第三个父代，如果为None则仅使用a和b
        eta: 探索参数，默认0.5。控制随机扰动的范围
        zeta: 开发参数，默认0.5。控制向父代回归的程度
        lower_bound: 变量下界
        upper_bound: 变量上界
        rng: 随机数生成器
    
    返回:
        (child1, child2): 两个子代
    
    示例:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> a = np.array([100, 50, 80, 120, 60, 90])
        >>> b = np.array([80, 60, 100, 100, 50, 70])
        >>> child1, child2 = pcx_crossover(a, b, eta=0.5, zeta=0.5,
        ...                               lower_bound=0, upper_bound=300, rng=rng)
    """
    # 参数校验
    if parent_a.shape != parent_b.shape:
        raise ValueError("父代维度必须一致")
    if parent_c is not None and parent_a.shape != parent_c.shape:
        raise ValueError("所有父代维度必须一致")
    if not (0 <= eta <= 1):
        raise ValueError("eta 必须在 [0, 1] 范围内")
    if not (0 <= zeta <= 1):
        raise ValueError("zeta 必须在 [0, 1] 范围内")
    
    if rng is None:
        rng = np.random.default_rng()
    
    dim = parent_a.shape[0]
    
    # 计算父代中心点
    if parent_c is not None:
        center = (parent_a + parent_b + parent_c) / 3.0
    else:
        center = (parent_a + parent_b) / 2.0
    
    # 生成两个子代
    child1 = np.zeros(dim)
    child2 = np.zeros(dim)
    
    for i in range(dim):
        # 计算父代a相对于中心的偏移
        diff_a = parent_a[i] - center[i]
        
        # 子代1: 以父代a为中心
        # child = a + zeta * (a - center) + eta * gaussian_noise
        # Fix: Noise should be proportional to parent distance, not global bounds
        dist_ab = abs(parent_a[i] - parent_b[i])
        noise1 = rng.normal(0, 1)
        child1[i] = parent_a[i] + zeta * diff_a + eta * noise1 * dist_ab
        
        # 子代2: 以父代b为中心
        diff_b = parent_b[i] - center[i]
        noise2 = rng.normal(0, 1)
        child2[i] = parent_b[i] + zeta * diff_b + eta * noise2 * dist_ab
    
    # 边界约束
    child1 = np.clip(child1, lower_bound, upper_bound)
    child2 = np.clip(child2, lower_bound, upper_bound)
    
    return child1, child2


def hybrid_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    population: np.ndarray | None = None,
    objective_values: np.ndarray | None = None,
    method: str = "sbx",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    混合交叉算子接口
    
    简介:
        提供统一的接口来调用不同的交叉算子。
        根据 method 参数选择具体的交叉策略。
    
    参数:
        parent_a: 第一个父代
        parent_b: 第二个父代
        population: 可选的种群数组（用于DE/PCX需要额外父代时）
        objective_values: 可选的目标值数组（用于选择最优父代）
        method: 交叉方法，可选 "sbx", "de", "pcx", "sbc"（原有）
        **kwargs: 传递给具体交叉算子的参数
    
    返回:
        (child1, child2): 两个子代
    
    示例:
        >>> child1, child2 = hybrid_crossover(a, b, method="sbx", eta=20,
        ...                                  lower_bound=0, upper_bound=300, rng=rng)
    """
    method = method.lower()
    
    if method == "adaptive_sbx":
        current_gen = kwargs.get("current_gen", 0)
        max_gen = kwargs.get("max_gen", 1)
        return adaptive_sbx_crossover(
            parent_a, parent_b,
            current_gen=current_gen,
            max_gen=max_gen,
            eta_start=kwargs.get("eta_start", 2.0),
            eta_end=kwargs.get("eta_end", 30.0),
            lower_bound=kwargs.get("lower_bound", 0.0),
            upper_bound=kwargs.get("upper_bound", 300.0),
            rng=kwargs.get("rng"),
        )
    elif method == "sbx":
        return sbx_crossover(
            parent_a, parent_b,
            eta=kwargs.get("eta", 20.0),
            lower_bound=kwargs.get("lower_bound", 0.0),
            upper_bound=kwargs.get("upper_bound", 300.0),
            rng=kwargs.get("rng"),
        )
    elif method == "de":
        # DE需要额外的父代
        if population is not None and len(population) >= 3:
            local_rng = kwargs.get("rng", np.random.default_rng())
            # 随机选择两个不同的父代
            indices = local_rng.choice(len(population), size=2, replace=False)
            parent_c = population[indices[0]]
            parent_d = population[indices[1]]
            return de_crossover(
                parent_a, parent_c, parent_d,
                cr=kwargs.get("cr", 0.9),
                f=kwargs.get("f", 0.5),
                lower_bound=kwargs.get("lower_bound", 0.0),
                upper_bound=kwargs.get("upper_bound", 300.0),
                rng=kwargs.get("rng"),
            )
        else:
            # 如果没有足够父代，回退到SBX
            return sbx_crossover(parent_a, parent_b, **kwargs)
    elif method == "pcx":
        # PCX可以使用第三个父代
        parent_c = kwargs.get("parent_c")
        return pcx_crossover(
            parent_a, parent_b, parent_c,
            eta=kwargs.get("eta", 0.5),
            zeta=kwargs.get("zeta", 0.5),
            lower_bound=kwargs.get("lower_bound", 0.0),
            upper_bound=kwargs.get("upper_bound", 300.0),
            rng=kwargs.get("rng"),
        )
    elif method == "sbc":
        # 调用原有的协同平衡交叉
        return synergistic_balance_crossover(
            parent_a, parent_b,
            stage_count=kwargs.get("stage_count", 4),
            alpha=kwargs.get("alpha", 0.5),
            lower_bound=kwargs.get("lower_bound", 0.0),
            upper_bound=kwargs.get("upper_bound", 300.0),
            rng=kwargs.get("rng"),
        )
    else:
        raise ValueError(f"未知的交叉方法: {method}")
