# --- 最终增强版SEM分析脚本 ---
# 目标：计算并输出论文所需的所有SEM拟合指标和参数

# 1. 准备工作
library(lavaan)
setwd("D:/AAA研究生学习/ER数据融合/全部实验代码") # 确保路径正确
df <- read.csv("preprocessed_data.csv")
df_fused <- read.csv("Mental_Health_Composite_Index.csv")
print("数据加载成功。")

# --------------------------------------------------------------------
# 模型一：参数估计模型 (饱和路径模型)
# --------------------------------------------------------------------
print("--- 正在运行模型一：饱和路径模型 ---")
path_model_spec <- '
  stress_level ~ Psychological_Score + 
                 Physical_Score + 
                 Environmental_Score + 
                 Academic_Score + 
                 Social_Score
'
fit_path <- sem(model = path_model_spec, data = df_fused, ordered = c("stress_level"))
path_params <- parameterEstimates(fit_path, standardized = TRUE)
significant_path_rules <- subset(path_params, op == "~" & pvalue < 0.05)

# 【新增】为论文生成参数估计表格
print("\n--- [论文用] 参数估计模型(饱和路径模型)的核心参数 ---")
parameter_table_data <- subset(significant_path_rules, select = c("rhs", "std.all", "pvalue"))
colnames(parameter_table_data) <- c("维度 (Predictor)", "标准化回归系数 (β)", "p值 (p-value)")
print(parameter_table_data)


# --------------------------------------------------------------------
# 模型二：理论验证模型 (最终单因子SEM)
# --------------------------------------------------------------------
print("\n--- 正在运行模型二：最终单因子SEM ---")
final_sem_spec <- '
  # 测量模型
  General_Adversity =~ anxiety_level + self_esteem + depression + 
                       academic_performance + study_load + teacher_student_relationship + future_career_concerns +
                       social_support + peer_pressure + extracurricular_activities + bullying +
                       headache + blood_pressure + sleep_quality + breathing_problem +
                       noise_level + living_conditions + safety + basic_needs
  # 结构模型
  stress_level ~ General_Adversity + mental_health_history
'
fit_final_sem <- sem(model = final_sem_spec, data = df, ordered = c("stress_level"))

# 【新增】为论文提取所有需要的拟合指标
print("\n--- [论文用] 理论验证模型(单因子SEM)的拟合指数 ---")
fit_measures_all <- fitMeasures(fit_final_sem)
# 提取所需指标
chi2 <- fit_measures_all['chisq.scaled']
df_val <- fit_measures_all['df.scaled']
chi2_df_ratio <- chi2 / df_val
cfi <- fit_measures_all['cfi.scaled']
tli <- fit_measures_all['tli.scaled']
rmsea <- fit_measures_all['rmsea.scaled']
rmsea_ci_lower <- fit_measures_all['rmsea.ci.lower.scaled']
rmsea_ci_upper <- fit_measures_all['rmsea.ci.upper.scaled']
srmr <- fit_measures_all['srmr_bentler'] # 使用 srmr_bentler 更常用

# 打印所有指标
cat(sprintf("卡方值 (Chi-square) = %.2f\n", chi2))
cat(sprintf("自由度 (df) = %d\n", df_val))
cat(sprintf("卡方自由度比 (χ²/df) = %.2f\n", chi2_df_ratio))
cat(sprintf("CFI = %.3f\n", cfi))
cat(sprintf("TLI = %.3f\n", tli))
cat(sprintf("RMSEA = %.3f\n", rmsea))
cat(sprintf("RMSEA 90%% CI = [%.3f, %.3f]\n", rmsea_ci_lower, rmsea_ci_upper))
cat(sprintf("SRMR = %.3f\n", srmr))

# ====================================================================
# 补充实验一：五因子CFA模型区分效度检验
# 目的: 检验五个核心维度作为独立潜变量的有效性，并评估其区分效度。
# ====================================================================

print("\n--- 正在运行补充实验一：五因子CFA模型区分效度检验 ---")

# 1. 定义五因子相关模型 (Confirmatory Factor Analysis, CFA)
# 根据各组数据融合脚本，确定每个潜变量(Latent Variable)对应的观测指标(Observed Indicators)
five_factor_model_spec <- '
  # 定义五个潜变量及其测量指标
  Psychological =~ anxiety_level + self_esteem + depression
  Physical      =~ headache + blood_pressure + sleep_quality + breathing_problem
  Environmental =~ noise_level + living_conditions + safety + basic_needs
  Academic      =~ academic_performance + study_load + teacher_student_relationship + future_career_concerns
  Social        =~ social_support + peer_pressure + extracurricular_activities + bullying
'

# 2. 拟合CFA模型
# 我们使用 cfa() 函数，它是 sem() 函数的一个便捷包装，专门用于拟合CFA模型
# 数据源应为预处理后的原始指标数据，即 preprocessed_data.csv
fit_five_factor <- cfa(model = five_factor_model_spec, data = df) # 确保 df 是从 preprocessed_data.csv 加载的

# 3. 提取并展示关键结果
# (a) 模型的整体拟合指数
print("\n--- [补充实验] 五因子CFA模型拟合指数 ---")
# 我们只关心几个核心指标
fit_measures_to_report <- c("chisq.scaled", "df.scaled", "cfi.scaled", "tli.scaled", "rmsea.scaled", "srmr")
fit_summary <- fitMeasures(fit_five_factor, fit_measures_to_report)
# 打印一个整洁的表格
print(round(fit_summary, 3))


# (b) 潜变量相关系数矩阵 (这是区分效度检验的核心！)
print("\n--- [补充实验] 潜变量相关系数矩阵 (用于检验区分效度) ---")
latent_correlations <- lavInspect(fit_five_factor, "cor.lv")
print(round(latent_correlations, 3))

# (c) 【可选】标准化因子载荷，检查每个指标是否有效地测量了其潜变量
# print("\n--- [补充实验] 标准化因子载荷 ---")
# print(standardizedSolution(fit_five_factor))

# 在 final_sem_analysis.R 脚本中新增此部分
install.packages(c("ggplot2", "ggExtra"))
library(ggplot2)
library(ggExtra)

print("\n--- 正在运行补充实验二：非线性关系可视化 ---")

# 使用已经融合了五个维度得分的数据框 df_fused
# 将整数型的stress_level稍微抖动一下，以便在散点图上更好地观察
df_fused$stress_level_jitter <- jitter(df_fused$stress_level, factor = 0.5)

# 以'Psychological_Score'为例
p <- ggplot(df_fused, aes(x = Psychological_Score, y = stress_level_jitter)) +
  geom_point(alpha = 0.2, color = "blue") +  # 半透明的散点
  geom_smooth(method = "loess", aes(y = stress_level), color = "red", se = TRUE) + # LOESS拟合曲线和置信区间
  labs(
    x = "Emotional Score (Higher = Worse)",
    y = "Stress Level (0=Low, 1=Mid, 2=High)"
  ) +
  theme_bw() # 使用SCI论文常用的黑白主题

# 添加边缘直方图/箱线图，以更清晰地展示数据分布
p_final <- ggMarginal(p, type = "histogram", fill="transparent")

# 保存高清图像
ggsave("non_linearity_visualization.png", plot = p_final, width = 8, height = 6, dpi = 300)
print("非线性关系图已生成并保存为 'non_linearity_visualization.png'")

# 你可以为 Academic_Score, Social_Score 等其他重要变量重复此过程