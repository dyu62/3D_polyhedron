blue = lambda x: '\033[94m' + x + '\033[0m'
red = lambda x: '\033[31m' + x + '\033[0m'
green = lambda x: '\033[32m' + x + '\033[0m'
yellow = lambda x: '\033[33m' + x + '\033[0m'
greenline = lambda x: '\033[42m' + x + '\033[0m'
yellowline = lambda x: '\033[43m' + x + '\033[0m'

valid_chars = 'EFHILOTUYZ'

alphabetic_labels = [char1 + char2 for char1 in valid_chars for char2 in valid_chars]
alphabetic_labels.sort()
label_mapping = {label: idx for idx, label in enumerate(alphabetic_labels)} # to number
reverse_label_mapping = {v: k for k, v in label_mapping.items()} # to alphabetic

single_alphabetic_labels=[char1 for char1 in valid_chars]
single_alphabetic_labels.sort()
single_label_mapping = {label: idx for idx, label in enumerate(single_alphabetic_labels)}
single_reverse_label_mapping = {v: k for k, v in single_label_mapping.items()}

shapenet_class_names= ["chair", "display",  
    "loudspeaker", "sofa", "table" ,"bathtub",  "bench", "bookshelf", "cabinet", "cellular telephone","file","knife","lamp","pot","vessel"]

modelnet_class_names = ["bed","bookshelf","chair","cone","desk","dresser","glass box","laptop",\
    "monitor","night stand","sofa","table","tent","tv stand"]

shapenet_class_to_int = {class_name: index for index, class_name in enumerate(shapenet_class_names)}
modelnet_class_to_int={class_name: index for index, class_name in enumerate(modelnet_class_names)}

shapenet_reverse = {v: k for k, v in shapenet_class_to_int.items()}
modelnet_reverse = {v: k for k, v in modelnet_class_to_int.items()}