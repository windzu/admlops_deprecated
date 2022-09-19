# Prerequisites

aaa

# Installation

> 只推荐docker安装

## 构建镜像

```bash
export MMDEPLOY_VERSION=0.7.0 && \
cd $MMLAB_EXTENSION_PATH/mmdeploy_extension && \
docker build docker/dev/ -t mmdeploy:$MMDEPLOY_VERSION \
--build-arg VERSION=$MMDEPLOY_VERSION \
--build-arg USE_SRC_INSIDE=true
```

## 创建容器

```bash
cd $MMLAB_EXTENSION_PATH/mmdeploy_extension && \
docker-compose up -d
```
