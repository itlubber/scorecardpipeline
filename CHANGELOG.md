# Changelog

## [Unreleased]

### Added

- **新模块 `scorecardpipeline.explainability`**：
  - 新增 `ScorecardExplainer` 类，支持对 WOE-LR 评分卡模型进行 SHAP 解释。
  - 支持批量解释、单样本解释、特征重要度汇总。
- **新模块 `scorecardpipeline.monitoring`**：
  - 新增 `ModelMonitor` 类，支持评分分布 PSI、特征 PSI、模型性能（KS/AUC）衰减监控。
  - 提供 `monitor_report` 方法生成完整监控报告。
- **新模块 `scorecardpipeline.calibration`**：
  - 新增 `ProbabilityCalibrator` 类，支持 Platt Scaling 和 Isotonic Regression 概率校准。
  - 兼容单概率、概率矩阵等多种输入形式。
- **测试与 CI/CD**：
  - 新增 `tests/` 目录及完整回归测试（`pytest`），覆盖 processing、scorecard、model、rule、explainability、monitoring、calibration 等核心模块。
  - 新增 `tox.ini`，通过 tox 统一管理测试和 lint 环境。
  - 新增 GitHub Actions 工作流（`.github/workflows/tests.yml`），使用 tox 在多平台/多 Python 版本下运行测试和 lint。
  - 新增 `.pre-commit-config.yaml`，集成 `ruff` 和 `black`。
- **演示 Notebook**：
  - 新增 `examples/advanced_features_demo.ipynb`，图文并茂展示 SHAP 解释、模型监控、概率校准功能，结果输出到 `examples/model_report/`。

### Fixed

- 修复 `Combiner.load()` 和 `WOETransformer.load()` 加载离线规则后未设置 `fitted_` 的问题。
- 修复 `scorecard.py` 在 `scikit-learn >= 1.6` 环境下 `check_array` 参数不兼容的问题。

### Changed

- `pyproject.toml` 依赖与原 `requirements.txt` 保持一致，不做过多拆分；仅 CairoSVG（需系统库）和 shap（新增功能）作为可选依赖。
- 更新 `README.md` 安装说明。
- 更新 `CLAUDE.md`，增加新模块架构说明。
