import pandas as pd

# 读取CSV文件
df = pd.read_csv('/mnt/nfs1/hanhc/NCU/data/ImageNet1K/validation/labels.csv')

# 替换image_path列中的路径
df['image_path'] = df['image_path'].str.replace('/mnt/hhc_data/', '/mnt/nfs1/hhc_data/')

# 保存修改后的CSV文件（覆盖原文件）
df.to_csv('/mnt/nfs1/hanhc/NCU/data/cc3m/train.csv', index=False)

# 如果你想保存到新文件，可以使用：
# df.to_csv('/mnt/nfs1/hanhc/NCU/data/cc3m/train_modified.csv', index=False)