# pairwise-tactics-zoc-counter

## 领域

游戏、图形与多媒体

## 主题

回合战棋 ZOC 与反击链

## 说明

ZOC 路径代价、反击链深度与去重、回合屏障。

初始状态下 `game/tactics_zoc.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
