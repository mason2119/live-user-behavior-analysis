# -*- coding: utf-8 -*-
"""
问题1：留存魔法数字分析
方法：逻辑回归 + MLP神经网络 + 贝叶斯优化调参 + 混淆矩阵 + 卡方检验
目标：找到影响次日留存的关键行为阈值（魔法数字）
"""

import os
import sys
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, roc_curve)
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

warnings.filterwarnings('ignore')

# 设置 joblib 临时目录为项目内纯英文路径，避免中文路径导致并行编码错误
os.environ['JOBLIB_TEMP_FOLDER'] = os.path.abspath('./joblib_temp')
os.makedirs(os.environ['JOBLIB_TEMP_FOLDER'], exist_ok=True)

# ============================================================
# 一、配置
# ============================================================
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_CODE_DIR)
DATA_DIR = os.path.join(_ROOT_DIR, 'data', '中文版')
OUTPUT_DIR = os.path.join(_CODE_DIR, 'output')
DOC_DIR = os.path.join(_ROOT_DIR, 'reports')
LOG_FILE = os.path.join(OUTPUT_DIR, '08_问题1_分析日志.txt')
DOC_FILE = os.path.join(DOC_DIR, '08_问题1_留存魔法数字分析报告.docx')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 二、日志工具：同时输出到终端和文件
# ============================================================
class Tee:
    """将stdout同时输出到终端和文件"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ============================================================
# 三、数据加载与清洗
# ============================================================
def load_and_clean_data():
    """加载用户行为日表和用户基础信息表，清洗异常值"""
    print("=" * 80)
    print("正在加载数据...")
    print("=" * 80)

    # 加载用户行为日表（核心表）
    behavior_path = os.path.join(DATA_DIR, '用户行为日表.csv')
    df_behavior = pd.read_csv(behavior_path)
    print(f"用户行为日表原始数据: {df_behavior.shape[0]:,} 行 × {df_behavior.shape[1]} 列")

    # 加载用户基础信息表
    user_path = os.path.join(DATA_DIR, '用户基础信息.csv')
    df_user = pd.read_csv(user_path)
    print(f"用户基础信息表原始数据: {df_user.shape[0]:,} 行 × {df_user.shape[1]} 列")

    # 数据清洗
    print("\n正在进行数据清洗...")

    # 1. 处理观看时长负值
    neg_duration = (df_behavior['总观看时长(秒)'] < 0).sum()
    print(f"  总观看时长(秒) 负值数量: {neg_duration}")
    df_behavior = df_behavior[df_behavior['总观看时长(秒)'] >= 0].copy()

    # 2. 处理年龄异常值（用户基础信息表）
    if '年龄' in df_user.columns:
        age_abnormal = ((df_user['年龄'] < 10) | (df_user['年龄'] > 80)).sum()
        print(f"  年龄异常值数量: {age_abnormal}")
        df_user.loc[(df_user['年龄'] < 10) | (df_user['年龄'] > 80), '年龄'] = np.nan

    # 3. 合并两张表
    df = df_behavior.merge(df_user, on='用户ID', how='left')
    print(f"\n合并后数据: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    print(f"合并后缺失值总数: {df.isnull().sum().sum()}")

    return df


# ============================================================
# 四、特征工程
# ============================================================
def feature_engineering(df):
    """构建留存预测特征"""
    print("\n" + "=" * 80)
    print("正在进行特征工程...")
    print("=" * 80)

    # 目标变量
    target = '次日是否留存'

    # 选择特征
    numeric_features = [
        '总观看时长(秒)',
        '点赞次数',
        '评论次数',
        '关注次数',
        '参与PK次数',
        '连续登录天数',
        '近7日平均观看时长',
        '时长变化率',
        '年龄',
        'VIP等级',
        '关注主播数',
        '社交强度得分',
    ]

    categorical_features = [
        '设备类型',
        '网络类型',
        '地区',
        '主要观看分类',
        '是否跨时区观看',
        '用户类型',
    ]

    # 衍生特征
    df['互动次数'] = df['点赞次数'] + df['评论次数'] + df['关注次数']
    df['互动密度'] = np.where(df['总观看时长(秒)'] > 0,
                              df['互动次数'] / df['总观看时长(秒)'], 0)
    df['观看时长分钟'] = df['总观看时长(秒)'] / 60

    numeric_features.extend(['互动次数', '互动密度', '观看时长分钟'])

    # 确保特征存在
    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]

    print(f"数值特征 ({len(numeric_features)}): {numeric_features}")
    print(f"分类特征 ({len(categorical_features)}): {categorical_features}")

    # 选择目标变量为次日是否留存的数据
    df_model = df[df[target].notna()].copy()
    print(f"\n用于建模的有效样本数: {len(df_model):,}")
    print(f"次日留存率: {df_model[target].mean():.2%}")

    # 由于数据量大，进行平衡采样以加速模型训练
    if len(df_model) > 400000:
        print("\n数据量较大，进行类别平衡采样...")
        retained = df_model[df_model[target] == 1]
        not_retained = df_model[df_model[target] == 0]

        # 每类最多取20万样本，保持类别平衡
        sample_size_per_class = 200000
        retained_sample = retained.sample(n=min(sample_size_per_class, len(retained)), random_state=42)
        not_retained_sample = not_retained.sample(n=min(sample_size_per_class, len(not_retained)), random_state=42)

        df_model = pd.concat([retained_sample, not_retained_sample], ignore_index=True)
        print(f"采样后样本数: {len(df_model):,}")
        print(f"采样后次日留存率: {df_model[target].mean():.2%}")

    X = df_model[numeric_features + categorical_features].copy()
    y = df_model[target].copy()

    return X, y, numeric_features, categorical_features, df_model


# ============================================================
# 五、数据划分与预处理
# ============================================================
def build_preprocessor(numeric_features, categorical_features):
    """构建数据预处理管道"""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor


def split_data(X, y):
    """划分训练集和测试集"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n训练集样本数: {len(X_train):,}")
    print(f"测试集样本数: {len(X_test):,}")
    print(f"训练集留存率: {y_train.mean():.2%}")
    print(f"测试集留存率: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test


# ============================================================
# 六、模型训练与评估
# ============================================================
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """评估模型并返回指标"""
    print(f"\n{'=' * 80}")
    print(f"模型评估: {model_name}")
    print(f"{'=' * 80}")

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 对于概率预测
    if hasattr(model, "predict_proba"):
        y_prob_train = model.predict_proba(X_train)[:, 1]
        y_prob_test = model.predict_proba(X_test)[:, 1]
    else:
        y_prob_train = y_pred_train.astype(float)
        y_prob_test = y_pred_test.astype(float)

    metrics = {
        'model_name': model_name,
        'train_auc': roc_auc_score(y_train, y_prob_train),
        'test_auc': roc_auc_score(y_test, y_prob_test),
        'train_acc': accuracy_score(y_train, y_pred_train),
        'test_acc': accuracy_score(y_test, y_pred_test),
        'test_precision': precision_score(y_test, y_pred_test),
        'test_recall': recall_score(y_test, y_pred_test),
        'test_f1': f1_score(y_test, y_pred_test),
        'confusion_matrix': confusion_matrix(y_test, y_pred_test),
        'classification_report': classification_report(y_test, y_pred_test, target_names=['未留存', '留存']),
        'y_prob_test': y_prob_test,
        'y_test': y_test,
    }

    print(f"训练集 AUC: {metrics['train_auc']:.4f}")
    print(f"测试集 AUC: {metrics['test_auc']:.4f}")
    print(f"测试集 准确率: {metrics['test_acc']:.4f}")
    print(f"测试集 精确率: {metrics['test_precision']:.4f}")
    print(f"测试集 召回率: {metrics['test_recall']:.4f}")
    print(f"测试集 F1值: {metrics['test_f1']:.4f}")
    print(f"\n混淆矩阵:")
    print(metrics['confusion_matrix'])
    print(f"\n分类报告:")
    print(metrics['classification_report'])

    return model, metrics


def train_logistic_regression(X_train, X_test, y_train, y_test, preprocessor):
    """训练逻辑回归模型"""
    print("\n" + "=" * 80)
    print("正在训练逻辑回归模型...")
    print("=" * 80)

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42, n_jobs=1))
    ])

    return evaluate_model(model, X_train, X_test, y_train, y_test, "逻辑回归")


def train_mlp_with_random_search(X_train, X_test, y_train, y_test, preprocessor):
    """使用随机搜索训练MLP神经网络并调参"""
    print("\n" + "=" * 80)
    print("正在使用随机搜索训练MLP神经网络...")
    print("=" * 80)

    # 先对数据预处理，便于MLP使用
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    param_dist = {
        'hidden_layer_sizes': [(64,), (128,), (128, 64)],
        'alpha': uniform(1e-4, 1e-2 - 1e-4),
        'learning_rate_init': uniform(1e-3, 1e-2 - 1e-3),
    }

    random_search = RandomizedSearchCV(
        MLPClassifier(max_iter=300, random_state=42, early_stopping=True),
        param_dist,
        n_iter=5,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=1,
        random_state=42,
        verbose=1
    )

    random_search.fit(X_train_processed, y_train)

    print(f"\n最优参数: {random_search.best_params_}")
    print(f"最优交叉验证 AUC: {random_search.best_score_:.4f}")

    best_model = random_search.best_estimator_

    # 评估最优模型
    y_pred_train = best_model.predict(X_train_processed)
    y_pred_test = best_model.predict(X_test_processed)
    y_prob_train = best_model.predict_proba(X_train_processed)[:, 1]
    y_prob_test = best_model.predict_proba(X_test_processed)[:, 1]

    metrics = {
        'model_name': 'MLP神经网络(随机搜索调参)',
        'best_params': str(random_search.best_params_),
        'cv_auc': random_search.best_score_,
        'train_auc': roc_auc_score(y_train, y_prob_train),
        'test_auc': roc_auc_score(y_test, y_prob_test),
        'train_acc': accuracy_score(y_train, y_pred_train),
        'test_acc': accuracy_score(y_test, y_pred_test),
        'test_precision': precision_score(y_test, y_pred_test),
        'test_recall': recall_score(y_test, y_pred_test),
        'test_f1': f1_score(y_test, y_pred_test),
        'confusion_matrix': confusion_matrix(y_test, y_pred_test),
        'classification_report': classification_report(y_test, y_pred_test, target_names=['未留存', '留存']),
        'y_prob_test': y_prob_test,
        'y_test': y_test,
    }

    print(f"训练集 AUC: {metrics['train_auc']:.4f}")
    print(f"测试集 AUC: {metrics['test_auc']:.4f}")
    print(f"测试集 准确率: {metrics['test_acc']:.4f}")
    print(f"\n混淆矩阵:")
    print(metrics['confusion_matrix'])
    print(f"\n分类报告:")
    print(metrics['classification_report'])

    return best_model, metrics


# ============================================================
# 七、魔法数字分析（阈值分析）
# ============================================================
def magic_number_analysis(df_model):
    """分析关键行为阈值与留存率的关系"""
    print("\n" + "=" * 80)
    print("正在分析留存魔法数字...")
    print("=" * 80)

    threshold_features = {
        '观看时长分钟': [1, 5, 10, 15, 20, 30, 45, 60],
        '点赞次数': [0, 1, 3, 5, 10, 20],
        '评论次数': [0, 1, 2, 3, 5],
        '互动次数': [0, 1, 3, 5, 10],
        '连续登录天数': [1, 2, 3, 5, 7],
    }

    magic_results = []

    for feature, thresholds in threshold_features.items():
        if feature not in df_model.columns:
            continue

        for threshold in thresholds:
            above = df_model[df_model[feature] >= threshold]
            below = df_model[df_model[feature] < threshold]

            if len(above) < 100 or len(below) < 100:
                continue

            retention_above = above['次日是否留存'].mean()
            retention_below = below['次日是否留存'].mean()

            # 卡方检验
            contingency = pd.crosstab(
                df_model[feature] >= threshold,
                df_model['次日是否留存']
            )
            chi2, p_value, _, _ = chi2_contingency(contingency)

            magic_results.append({
                '特征': feature,
                '阈值': threshold,
                '大于等于阈值样本数': len(above),
                '大于等于阈值留存率': retention_above,
                '小于阈值留存率': retention_below,
                '留存率差异': retention_above - retention_below,
                '卡方值': chi2,
                'P值': p_value,
                '显著': '是' if p_value < 0.05 else '否'
            })

    magic_df = pd.DataFrame(magic_results)
    magic_df = magic_df.sort_values('留存率差异', ascending=False)

    print("\nTop 10 魔法数字（按留存率差异排序）:")
    print(magic_df.head(10).to_string(index=False))

    # 保存到CSV
    magic_path = os.path.join(OUTPUT_DIR, '08_留存魔法数字分析结果.csv')
    magic_df.to_csv(magic_path, index=False, encoding='utf-8-sig')
    print(f"\n魔法数字结果已保存: {magic_path}")

    return magic_df


# ============================================================
# 八、可视化
# ============================================================
def plot_confusion_matrix(metrics, model_name, save_path):
    """绘制混淆矩阵热力图"""
    cm = metrics['confusion_matrix']

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=[0, 1],
           yticks=[0, 1],
           xticklabels=['未留存', '留存'],
           yticklabels=['未留存', '留存'],
           title=f'{model_name} 混淆矩阵',
           ylabel='真实标签',
           xlabel='预测标签')

    # 在格子里显示数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"混淆矩阵图已保存: {save_path}")


def plot_roc_curves(metrics_list, save_path):
    """绘制多个模型的ROC曲线对比"""
    fig, ax = plt.subplots(figsize=(8, 6))

    for metrics in metrics_list:
        fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_prob_test'])
        auc = metrics['test_auc']
        ax.plot(fpr, tpr, label=f"{metrics['model_name']} (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], 'k--', label='随机猜测')
    ax.set_xlabel('假阳性率 (False Positive Rate)')
    ax.set_ylabel('真阳性率 (True Positive Rate)')
    ax.set_title('留存预测模型 ROC 曲线对比')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"ROC曲线图已保存: {save_path}")


def plot_magic_number(magic_df, save_path):
    """绘制魔法数字留存率差异图"""
    top_magic = magic_df.head(15)

    fig, ax = plt.subplots(figsize=(12, 8))
    labels = [f"{row['特征']} >= {row['阈值']}" for _, row in top_magic.iterrows()]
    values = top_magic['留存率差异'].values

    colors = ['#2E86AB' if v > 0 else '#A23B72' for v in values]
    bars = ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('留存率差异')
    ax.set_title('留存魔法数字分析（Top 15 阈值效应）')
    ax.grid(True, axis='x', alpha=0.3)

    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val, i, f' {val:.2%}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"魔法数字图已保存: {save_path}")


# ============================================================
# 九、生成Word报告
# ============================================================
def create_word_report(metrics_list, magic_df):
    """生成Word分析报告"""
    print("\n" + "=" * 80)
    print("正在生成Word分析报告...")
    print("=" * 80)

    doc = Document()

    # 标题
    title = doc.add_heading('问题1：留存魔法数字分析报告', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 分析时间
    doc.add_paragraph(f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph()

    # 一、分析目标
    doc.add_heading('一、分析目标', level=1)
    doc.add_paragraph(
        '本分析旨在识别影响用户次日留存的关键行为阈值（即"魔法数字"），'
        '并通过逻辑回归与MLP神经网络模型预测用户留存概率，为产品运营提供数据支持。'
    )

    # 二、模型评估结果
    doc.add_heading('二、模型评估结果', level=1)

    table = doc.add_table(rows=1, cols=7)
    table.style = 'Light Grid Accent 1'

    hdr_cells = table.rows[0].cells
    headers = ['模型', '测试集AUC', '测试集准确率', '精确率', '召回率', 'F1值', '训练集AUC']
    for i, h in enumerate(headers):
        hdr_cells[i].text = h

    for metrics in metrics_list:
        row_cells = table.add_row().cells
        row_cells[0].text = metrics['model_name']
        row_cells[1].text = f"{metrics['test_auc']:.4f}"
        row_cells[2].text = f"{metrics['test_acc']:.4f}"
        row_cells[3].text = f"{metrics['test_precision']:.4f}"
        row_cells[4].text = f"{metrics['test_recall']:.4f}"
        row_cells[5].text = f"{metrics['test_f1']:.4f}"
        row_cells[6].text = f"{metrics['train_auc']:.4f}"

    doc.add_paragraph()

    # 三、混淆矩阵可视化
    doc.add_heading('三、混淆矩阵可视化', level=1)
    for metrics in metrics_list:
        doc.add_paragraph(f"{metrics['model_name']} 混淆矩阵:")
        img_path = os.path.join(OUTPUT_DIR, f'08_{metrics["model_name"]}_混淆矩阵.png')
        if os.path.exists(img_path):
            doc.add_picture(img_path, width=Inches(5.0))
            last_paragraph = doc.paragraphs[-1]
            last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 四、ROC曲线对比
    doc.add_heading('四、ROC曲线对比', level=1)
    roc_path = os.path.join(OUTPUT_DIR, '08_ROC曲线对比.png')
    if os.path.exists(roc_path):
        doc.add_picture(roc_path, width=Inches(5.5))
        last_paragraph = doc.paragraphs[-1]
        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 五、留存魔法数字分析
    doc.add_heading('五、留存魔法数字分析', level=1)
    doc.add_paragraph(
        '下表展示了不同行为阈值对次日留存率的影响。'
        '卡方检验用于验证阈值效应的统计显著性（P < 0.05 认为显著）。'
    )

    # 添加Top 10魔法数字表格
    top10 = magic_df.head(10)
    table2 = doc.add_table(rows=1, cols=6)
    table2.style = 'Light Grid Accent 1'
    hdr_cells2 = table2.rows[0].cells
    headers2 = ['特征', '阈值', '样本数', '留存率(>=阈值)', '留存率(<阈值)', 'P值']
    for i, h in enumerate(headers2):
        hdr_cells2[i].text = h

    for _, row in top10.iterrows():
        row_cells = table2.add_row().cells
        row_cells[0].text = str(row['特征'])
        row_cells[1].text = str(row['阈值'])
        row_cells[2].text = f"{row['大于等于阈值样本数']:,}"
        row_cells[3].text = f"{row['大于等于阈值留存率']:.2%}"
        row_cells[4].text = f"{row['小于阈值留存率']:.2%}"
        row_cells[5].text = f"{row['P值']:.2e}"

    doc.add_paragraph()

    # 魔法数字图
    magic_path = os.path.join(OUTPUT_DIR, '08_魔法数字效应图.png')
    if os.path.exists(magic_path):
        doc.add_picture(magic_path, width=Inches(5.5))
        last_paragraph = doc.paragraphs[-1]
        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 六、结论与建议
    doc.add_heading('六、结论与建议', level=1)
    best_model = max(metrics_list, key=lambda x: x['test_auc'])
    doc.add_paragraph(f"1. 最优模型为 {best_model['model_name']}，测试集 AUC 为 {best_model['test_auc']:.4f}。")

    if not magic_df.empty:
        top1 = magic_df.iloc[0]
        doc.add_paragraph(
            f"2. 最重要的魔法数字: {top1['特征']} >= {top1['阈值']}，"
            f"可使次日留存率提升 {top1['留存率差异']:.2%} (P={top1['P值']:.2e})。"
        )

    doc.add_paragraph(
        '3. 建议产品侧针对达到魔法数字阈值的用户设计激励策略，'
        '如对观看超过关键时长的用户推送个性化内容，提升整体留存。'
    )

    # 保存
    doc.save(DOC_FILE)
    print(f"Word报告已保存: {DOC_FILE}")


# ============================================================
# 十、主程序
# ============================================================
def main():
    # 重定向输出到日志文件
    sys.stdout = Tee(LOG_FILE)

    print("\n" + "=" * 80)
    print("问题1：留存魔法数字分析")
    print("=" * 80)
    print(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 加载数据
    df = load_and_clean_data()

    # 2. 特征工程
    X, y, numeric_features, categorical_features, df_model = feature_engineering(df)

    # 3. 数据划分
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 4. 构建预处理管道
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # 5. 训练逻辑回归
    lr_model, lr_metrics = train_logistic_regression(
        X_train, X_test, y_train, y_test, preprocessor
    )

    # 6. 训练MLP（随机搜索调参）
    mlp_model, mlp_metrics = train_mlp_with_random_search(
        X_train, X_test, y_train, y_test, preprocessor
    )

    # 7. 魔法数字分析
    magic_df = magic_number_analysis(df_model)

    # 8. 可视化
    metrics_list = [lr_metrics, mlp_metrics]

    plot_confusion_matrix(lr_metrics, lr_metrics['model_name'],
                          os.path.join(OUTPUT_DIR, f'08_{lr_metrics["model_name"]}_混淆矩阵.png'))
    plot_confusion_matrix(mlp_metrics, mlp_metrics['model_name'],
                          os.path.join(OUTPUT_DIR, f'08_{mlp_metrics["model_name"]}_混淆矩阵.png'))
    plot_roc_curves(metrics_list, os.path.join(OUTPUT_DIR, '08_ROC曲线对比.png'))
    plot_magic_number(magic_df, os.path.join(OUTPUT_DIR, '08_魔法数字效应图.png'))

    # 9. 生成Word报告
    create_word_report(metrics_list, magic_df)

    print("\n" + "=" * 80)
    print("问题1分析完成！")
    print(f"结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
