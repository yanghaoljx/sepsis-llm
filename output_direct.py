from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
import pandas as pd
from tqdm import tqdm

# Define the models (though you are only using qwen2 here)
models = ['gemma', 'llama3', 'qwen2', 'llama3.1', 'mistral']

# Load the LLM with the qwen2 model
llm = Ollama(model="qwen2", temperature=0.3)

# Define the function to score symptoms
def score_symptoms(llm, context_str):
    prompt_template = """
    Given a patient's medical record, predict the likelihood of in-hospital mortality as a single number.
    Return only a probability score between 0 and 1, where 0 represents no likelihood of death and 1 represents certainty of death.
    Do not provide any explanation or additional information—only the number.
    Medical Record: {medical_record}
    """

    # Create the prompt and chain
    prompt = PromptTemplate(template=prompt_template, input_variables=["medical_record"])
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.invoke({"medical_record": context_str})

    # Print result to inspect structure
    print(result)

    # Extract the score from the result (assuming it's in 'text')
    if 'text' in result:
        try:
            score = float(result['text'].strip())  # Convert the result to float
        except ValueError:
            score = None  # Handle any conversion errors
    else:
        score = None

    return score

# Load the input Excel file
df = pd.read_excel('首次病程v2.xlsx')

# Prepare a list to store output rows
output_rows = []

# Iterate through each row in the DataFrame
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
    patient_id = row['病案号']  # Extract the patient ID
    context_str = f"主诉: {row['主诉']}, {row['主要症状']}"  # Create the context string

    try:
        # Get the prediction score from the LLM
        result = score_symptoms(llm, context_str)
        print(f"Patient ID: {patient_id}, Score: {result}")

        # Append the result to the output rows
        output_rows.append({'病案号': patient_id, 'Mortality Score': result})
    
    except Exception as e:
        print(f"Error processing patient {patient_id}: {e}")
        # Append None or some error indication if there's an issue
        output_rows.append({'病案号': patient_id, 'Mortality Score': None})


# Create a DataFrame from the results
output_df = pd.DataFrame(output_rows)

# Merge the original DataFrame with the new scores
final_df = pd.merge(df, output_df, on='病案号')

# Save the merged DataFrame to a new Excel file
final_df.to_excel('mortality_predictions.xlsx', index=False)

print("Results have been saved to 'mortality_predictions.xlsx'.")


import pandas as pd 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score, precision_recall_fscore_support
df1 = pd.read_excel('qwen2合并检验.xlsx')[['就诊号','Label']]
df2 = pd.read_excel('mortality_predictions.xlsx')[['就诊号','Mortality Score']]
df = pd.merge(df1,df2,on='就诊号')


labels = df['Label']
scores = df['Mortality Score']

# 生成ROC曲线数据
fpr, tpr, thresholds = roc_curve(labels, scores)
roc_auc = roc_auc_score(labels, scores)

# 找到距离 (0, 1) 最近的点作为最优阈值
distances = np.sqrt(fpr**2 + (1 - tpr)**2)
best_threshold_index = np.argmin(distances)
best_threshold = thresholds[best_threshold_index]

# 根据最优阈值计算预测值
predictions = np.where(scores >= best_threshold, 1, 0)
accuracy = accuracy_score(labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')

# 打印结果
print(f'最优阈值: {best_threshold:.2f}')
print(f'准确率: {accuracy:.2f}')
print(f'F1分数: {f1:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'ROC AUC: {roc_auc:.2f}')

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.scatter(fpr[best_threshold_index], tpr[best_threshold_index], color='red', label=f'Best Threshold = {best_threshold:.2f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid(True) # 显示网格线
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()