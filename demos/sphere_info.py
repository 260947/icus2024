import bpy
import json

# 遍历场景中的所有对象
path = []
for obj in bpy.context.scene.objects:
    # 检查对象名字中是否包含'Sphere'
    if "球体" in obj.name:
        # 获取对象的位置
        location = list(obj.location)
        # 假设球体没有被非均匀缩放，我们可以取scale的X分量作为半径的估计值
        # 注意：这只在球体保持均匀缩放时有效
        radius = obj.scale[0]  # 使用对象的scale和dimension来估计半径
        path.append(location + [radius])
        print(f"球体名: {obj.name}, 位置: {location}, 半径: {radius}")

filename = '../viz_utils/sphere_info.json'
pose_dict = {}
with open(filename, "w+") as f:
    pose_dict["loc"] = path
    json.dump(pose_dict, f, indent=4)
