# 根据实际 UV 修正四角接缝

此步骤用于四面已经完成基本校色、贴回模型仍有硬切的情况。保留远离缝的材质感；不要靠继续全图压平、改 UV 或重画细纹消除一条局部色阶。

## 先定位真实原因

通过用户授权的 Blender 连接只读取得当前编辑状态的网格、每面材质和逐角 UV。编辑模式使用 BMesh 当前数据，不能只读可能滞后的 object.data。记录图像的实际文件及哈希，并核对节点中的 UV 层、Mapping、色彩空间、金属度、粗糙度、法线和边平滑状态。文件若被用户移动，以当前文件及内容核实，不能按旧文件名猜测。

本脚本假定每个材质有一个基础色图像、UV Mapping 恒等。有旋转/缩放等节点变换时，先将变换准确纳入采样坐标；不直接使用未变换 UV。材质或法线有差异时应先区分原因，不能默认以贴图补偿所有渲染差异。

使用无光照/Emission 预览隔离贴图本身的断层，再用相同灯光和 PBR 参数前后对照。后台场景副本可用于验证；不要覆盖用户原工程。需要切换用户当前预览时保留旧图像绑定，便于回退。

## 建立完整配对

UV 接缝往往在照片内部。像素中心换算为 `x=u*width-0.5`、`y=(1-v)*height-0.5`，不是沿 PNG 外轮廓取样。

使用 [../scripts/extract_uv_seams.py](../scripts/extract_uv_seams.py) 从网格导出数据提取配对曲线。它检查共享的跨材质边，也检查不同顶点索引、不同分段的共线开放边之精确重叠。T 形分段不一致可能让仅依赖共享边的算法漏掉接缝中央部分。必须核对世界空间总弧长、分段覆盖和同材质 UV 接点连续性，不能凭外观补齐缺口。分叉、重复或 UV 不连续时应拆成明确的独立区域。

输入 scene JSON 至少包含：

- `vertices`: `{id, co:[x,y,z]}`；`matrix_world`: 4×4对象变换。
- `faces`: `{id, material:材质索引, vertices:[顶点id…], uv:[[u,v]…]}`，顶点与 UV 顺序对应。
- `edges`: `{id, verts:[id,id], faces:[相邻面id…]}`。
- `materials`: `{name, nodes:[{type:"ShaderNodeTexImage", path:图像绝对路径}]}`。

提取命令：

```bash
python3 scripts/extract_uv_seams.py --scene /绝对路径/scene.json --output /绝对路径/seam-chains.json
```

输出的每条 `chains` 有同一三维弧长参数 `arc` 和两个 `sides`。每侧有像素曲线 `points`、朝各自材质内部的像素方向 `normals`、`px_per_unit`。提取器按对象变换后的坐标计算长度，过渡宽度必须适应当前模型单位。不能把一次样本的 `.20` 当作通用毫米值。

## 两侧共同校正

[../scripts/match_uv_seams.py](../scripts/match_uv_seams.py) 从两侧相同三维位置的低频颜色求共享边界剖面，同时调节跨缝梯度。立方过渡在远端回到原图、校正导数归零；少量 UV 边外延续用于滤波采样，不改变 alpha 或照片尺寸。原图细节残差继续使用源像素，不复制另一面的纹理。

任务 JSON：

```json
{
  "scene": "/绝对路径/records/scene.json",
  "seam_chains": "/绝对路径/records/seam-chains.json",
  "output_dir": "/绝对路径/新版/images",
  "records_dir": "/绝对路径/新版/records",
  "width_units": 0.20,
  "sigma": 3,
  "derivative_step_units": 0.015,
  "guard_pixels": 6,
  "views": {
    "top": {
      "source": "/绝对路径/当前已优化/top.png",
      "editable_mask": "/绝对路径/原校色范围/top-editable.png",
      "protected_mask": "/绝对路径/原保护范围/top-protected.png"
    }
  }
}
```

`views` 必须包含全部配对材质；示例只展示一面的字段。`seam_chains` 可省略，此时从 `scene` 完整提取配对。先确认 mask 与当前图片尺寸/内容一致。`sigma` 是低频采样的像素尺度；`width_units`、导数采样步距是模型世界坐标单位；调节前核对实际像素宽度及保护区距离。还可设 `max_lab_shift`（默认 `[18,12,12]`）、`max_slope_effect`（默认 `[16,10,10]`）、`protection_feather`（默认3像素）。

```bash
python3 scripts/match_uv_seams.py --job /绝对路径/records/job.json
```

脚本拒绝覆盖已有成品；记录目录必须在四面成品目录之外。保存有效接缝蒙版、源哈希和同采样方式的前后 ΔE76。指标仅统计两边都可完整校正的样本，须同时报告样本数量；受保护边缘和材质细纹仍需视觉检查。尺寸、alpha、保护区、接缝范围外 RGBA 必须不变。

## 验收

四个角都需要检查，尤其是中央大面和前后倒角交界。检查 C0 色阶是否消失、渐变是否自然、是否引入亮圈/暗带，再检查粒纹密度差。低频颜色指标不能证明高频纹理完全连续；保留可接受的实拍纹理差异，不能为了统计数值而抹掉瑕疵或添加新纹理。描述结果为已验证的具体改善，不承诺任意灯光/角度下绝对无缝。
