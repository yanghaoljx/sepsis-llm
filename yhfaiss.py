from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

def read_external_knowledge(path):
    path = './exsit_knowledge/knowledge.pkl'
    with open(path, 'rb') as file:
        loaded_data = pickle.load(file)
    paragraph = [loaded_data[i] for i in range(len(loaded_data))]
    return paragraph

def read_reports(path):
    reports = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            filepath = os.path.join(path, filename)
            with open(filepath, 'r') as f:
                txt = f.read()
                reports.extend(txt.split('\n'))
    return reports

def retrieval_info(reports, path, k):
    paragraphs = read_external_knowledge(path + '/exsit_knowledge')
    model = SentenceTransformer('all-mpnet-base-v2')
    p_embeddings = model.encode(paragraphs, convert_to_numpy=True)
    report_embeddings = model.encode(reports, convert_to_numpy=True)
    # Initialize FAISS index
    dim = p_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(p_embeddings)
    # Perform search
    distances, indices = index.search(report_embeddings, k)

    # Collect results
    knowledge = []
    for i in range(len(reports)):
        for idx in indices[i]:
            knowledge.append(paragraphs[idx])
    
    knowledge = list(set(knowledge))  # Remove duplicates
    return knowledge

if __name__ == '__main__':
    reports = read_reports(
        './dataset_folder/health_report_{10}')  # 13452
    print(len(reports))
    know = retrieval_info(reports, './', 3)
    for i in know:
        print(i)



# import pickle
# 要序列化的文本文件路径
# txt_file_path = './exsit_knowledge/knowledge_translated.txt'
# 要保存的Pickle文件路径
# pkl_file_path =  './exsit_knowledge/knowledge.pkl'
# 读取文本文件内容
# paragraphs = []
# with open(txt_file_path, 'r', encoding='utf-8') as file:
#     for line in file:
#         可以在这里添加逻辑来处理每一行，例如检查是否是段落的开始或结束
#         假设每行是一个段落的开始
#         paragraphs.append(line.strip())  # 使用strip()去除行尾的换行符
# 序列化文本数据并保存到Pickle文件
# with open(pkl_file_path, 'wb') as pkl_file:
#     pickle.dump(paragraphs, pkl_file)
# print(f"文本数据已保存到 {pkl_file_path}")