## `scorecardpipeline` 简介

`scorecardpipeline` 封装了 `toad`、`scorecardpy`、`optbinning` 等评分卡建模相关组件，`API` 风格与 `sklearn` 高度一致，支持 `pipeline` 式端到端评分卡建模、模型报告输出、导出 `PMML` 文件、超参数搜索等

> 在线文档: [`https://itlubber.github.io/scorecardpipeline-docs`](https://itlubber.github.io/scorecardpipeline-docs/)
> 
> pipy包: [`https://pypi.org/project/scorecardpipeline`](https://pypi.org/project/scorecardpipeline/)
>
> 仓库地址: [`https://github.com/itlubber/scorecardpipeline`](https://github.com/itlubber/scorecardpipeline)


## 版本更新说明

| **序号** | **版本号** | **发布日期** |                                  **版本说明**                                  |
| :------: | :--------: | :----------: | :----------------------------------------------------------------------------: |
|    01    |   0.1.0    |  2023.05.04  |                                     初始化                                     |
|    ..    |   ......   |    ......    |                                     ......                                     |
|    26    |   0.1.26   |  2023.11.13  | 稳定版本，新增方法[`说明文档`](https://itlubber.github.io/scorecardpipeline-docs/) |


## 环境安装准备

1. 安装 `python` 环境

推荐 `python 3.8.13` ，参照网上教程自行安装即可，也可直接使用系统提供的 `python3` , 版本不易过高, 最低支持 `python 3.6`，最高支持 `python 3.10`

2. 安装 `scorecardpipeline` 包

+ 在线环境安装

直接通过 `pip` 安装即可，安装后需要确认下版本是否安装正确，推荐 `-i https://test.pypi.org/simple` 指定源安装

```shell
pip install scorecardpipeline
```

+ 离线环境安装

离线环境安装需先找一个 `python` 版本和系统 (`windows`、`linux`、`mac`) 与生产一致的有网机器或者容器，在有网的环境中下载相关依赖项后拷贝到离线环境安装

```shell
# 在线环境依赖下载
pip download -d site-packages/ scorecardpipeline
# 离线环境
pip install --no-index --find-links=site-packages scorecardpipeline
```

3. 安装 `jdk 1.8+` [可选]

如果有将训练完成的评分卡模型直接导出 `PMML` 模型文件部署的需求，需要单独安装 `java` 环境

> `mac` & `windows` 参考: [`https://developer.aliyun.com/article/1082599`](https://developer.aliyun.com/article/1082599)
>
> `linux` 参考: [`https://blog.csdn.net/muzi_gao/article/details/132169159`](https://blog.csdn.net/muzi_gao/article/details/132169159)


## 代码使用示例


