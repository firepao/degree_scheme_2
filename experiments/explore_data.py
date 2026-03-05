import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_and_explore_data(file_path):
    print(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    # 1. 基本信息
    print("\n" + "="*50)
    print("数据集基本信息")
    print("="*50)
    print(f"数据形状 (Shape): {df.shape}")
    print("\n列名 (Columns):")
    print(df.columns.tolist())
    print("\n数据类型 (Dtypes):")
    print(df.dtypes)
    
    print("\n" + "="*50)
    print("缺失值检查")
    print("="*50)
    missing = df.isnull().sum()
    print(missing[missing > 0])
    if missing.sum() == 0:
        print("未发现缺失值。")

    # 2. 关键变量统计 (N, P, K, Yield)
    key_vars = ['Nitrogen', 'Phosphorus', 'Potassium', 'Yield']
    # 检查列是否存在
    existing_vars = [col for col in key_vars if col in df.columns]
    
    print("\n" + "="*50)
    print("关键变量描述性统计")
    print("="*50)
    print(df[existing_vars].describe())

    # 3. 可视化分布
    print("\n正在生成可视化图表...")
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(existing_vars):
        plt.subplot(2, 2, i+1)
        sns.histplot(df[col], kde=True)
        plt.title(f'{col} 分布')
        plt.xlabel(col)
        plt.ylabel('频数')
    plt.tight_layout()
    plt.savefig('data_distribution.png')
    print("已保存分布图至 data_distribution.png")

    # 4. 相关性分析
    plt.figure(figsize=(10, 8))
    # 只计算数值列的相关性
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    # 重点关注与 Yield 的相关性
    if 'Yield' in corr.index:
        print("\n" + "="*50)
        print("与 Yield (产量) 的相关性 (Top 10)")
        print("="*50)
        print(corr['Yield'].sort_values(ascending=False).head(10))
    
    sns.heatmap(corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('特征相关性热力图')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    print("已保存相关性热力图至 correlation_heatmap.png")

    # 5. 季节 (Season) 分析 - 检查是否可作为阶段
    if 'Season' in df.columns:
        print("\n" + "="*50)
        print("Season (季节) 分布")
        print("="*50)
        print(df['Season'].value_counts())
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Season', y='Yield', data=df)
        plt.title('不同季节下的产量分布')
        plt.savefig('season_yield_boxplot.png')
        print("已保存季节产量箱线图至 season_yield_boxplot.png")

    # 6. 建议与下一步
    print("\n" + "="*50)
    print("数据洞察与建议")
    print("="*50)
    print("1. 数据量: ", df.shape[0], "条样本，足够支持机器学习建模。")
    if 'Season' in df.columns:
        seasons = df['Season'].unique()
        print(f"2. 阶段划分: 数据包含 {len(seasons)} 个季节 ({', '.join(map(str, seasons))})。")
        print("   - 方案一: 直接使用 Season 作为 categorical 特征。")
        print("   - 方案二 (推荐): 将不同 Season 视为同一作物的不同生长阶段（尽管实际上可能是不同批次），\n"
              "     构建多阶段优化问题。例如：Spring -> Summer -> Autumn (如果存在)。")
    else:
        print("2. 阶段划分: 未发现显式的时间/阶段列，可能需要人工构造阶段。")

if __name__ == "__main__":
    # 使用相对路径，假设在项目根目录下运行
    data_path = Path("data/raw/strawberry_nutrients.csv")
    if not data_path.exists():
        # 尝试使用绝对路径作为后备
        # 注意：这里硬编码了路径，实际项目中最好用配置
        import os
        current_dir = os.getcwd() # D:\degree_code_scheml_2\scheml_2
        data_path = Path(current_dir) / "data/raw/strawberry_nutrients.csv"
        
    load_and_explore_data(data_path)
