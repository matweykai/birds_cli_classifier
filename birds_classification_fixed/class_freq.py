import os


train_dir = './train'
class_dir_path_list = [os.path.join(train_dir, name) 
		       for name in os.listdir(train_dir)]

res_dict = { class_dir_path: len(os.listdir(class_dir_path)) 
	   for class_dir_path in class_dir_path_list if os.path.isdir(class_dir_path)}

dataset_size = sum(res_dict.values())

for dir_name, count in sorted(res_dict.items(), key=lambda x: x[1]):
	print(f'{dir_name} -> {count}({round(count / dataset_size * 100, 1)})%')

