from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from yhfaiss import retrieval_info
import json



llm = Ollama(model="qwen2")
def get_systemprompt_template(exist_context):

    chat_text_qa_msgs = [
        SystemMessagePromptTemplate.from_template(
            exist_context
        ),
        HumanMessagePromptTemplate.from_template(
            "请以标准的JSON格式直接提供结果，不包含任何其他解释、转义符、换行符或反斜杠，以及任何形式的开头，包括一个从0到1的分数，以及带有'score:'和'reason:'的评估依据\n"
            "'score': \n"
            "'reason': \n"
            "分数必须是0-1之间的数值\n"
            "这是答案的规则：未明确提起的症状请将score直接置为0，0-0.2是轻微或无，0.3-0.6是中等，0.7以上是严重\n"
            "这是患者的医疗记录。以下是上下文信息\n"
            "---------------------\n"
            "{context_str}"
            "根据上下文信息，你是一个乐于助人的健康管理顾问"
            "回答下面问题: {query_str}\n"
        )
    ]
    chat_text_qa_msgs_lc = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
    return chat_text_qa_msgs_lc

def get_systemprompt_template_without_reason(exist_context):

    chat_text_qa_msgs = [
        SystemMessagePromptTemplate.from_template(
            exist_context
        ),
        HumanMessagePromptTemplate.from_template(
            "请以标准的json格式直接给出答案，不包含任何其他解释、转义符、换行符或反斜杠，以及任何形式的开头，其中只有一个介于0和1之间的数字：\n"
            "'score': \n"
            "分数必须是0-1之间的数值\n"
            "这是答案的规则：未明确提起的症状请将score直接置为0，0-0.2是轻微或无，0.3-0.6是中等，0.7以上是严重\n"
            "这是患者的医疗记录。以下是上下文信息\n"
            "---------------------\n"
            "{context_str}"
            "根据上下文信息，你是一个症状评估的临床顾问"
            "回答下面问题: {query_str}\n"
        )
    ]
    chat_text_qa_msgs_lc = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
    return chat_text_qa_msgs_lc

def llm_predict_score(llm,syptom,context_str):
    sentence = ["案例中描述的人是否有{0}症状？你认为这种情况严重吗?请根据事实情况给出症状的评分，未明确提起的症状请将score直接置为0".format(syptom)]
    retrieval = retrieval_info(sentence, './', 1)
    related_work = [] 
    related_work.append(retrieval[0])
    context = 'The context provided here includes additional specialized health knowledge that can assist you in more effectively analyzing the reports.'
    my_context = context +  related_work[0]
    QA_PROMPT = get_systemprompt_template(my_context)
    formatted_prompt =QA_PROMPT.format_messages(
            context_str=context_str,
            query_str=sentence[0]
        )
    response = llm.invoke(formatted_prompt)
    print(response)
    try:
        # 尝试解析 JSON
        temp = json.loads(response)
        # 验证 JSON 结构
        if 'score' in temp and 'reason' in temp:
            score = temp['score']
            reason = temp['reason']
            return score,reason
        else:
            raise ValueError("JSON 结构有误'")
    except json.JSONDecodeError as e:
        # JSON 解析错误
        print(f"解析 JSON 时发生错误: {e}")
        return None, None
    except ValueError as ve:
        # 结构验证错误
        print(ve)
        return None, None
    

def llm_predict_score_without_reason(llm,syptom,context_str):
    sentence = ["案例中描述的人是否有{0}症状？你认为这种情况严重吗?请根据事实情况给出症状的评分，未明确提起的症状请直接置为0".format(syptom)]
    retrieval = retrieval_info(sentence, './', 1)
    related_work = [] 
    related_work.append(retrieval[0])
    context = '这里有一些额外的专业健康知识，可以帮助你更好地分析报告'
    my_context = context +  related_work[0]
    QA_PROMPT = get_systemprompt_template_without_reason(my_context)


    formatted_prompt =QA_PROMPT.format_messages(
            context_str=context_str,
            query_str=sentence[0]
        )
    response = llm.invoke(formatted_prompt)
    print(response)
    try:
        # 尝试解析 JSON
        temp = json.loads(response)
        # 验证 JSON 结构
        if 'score' in temp:
            score = temp['score']
            return score
        else:
            raise ValueError("JSON 结构不包含 'score'")
    except json.JSONDecodeError as e:
        # JSON 解析错误
        print(f"解析 JSON 时发生错误: {e}")
    except ValueError as ve:
        # 结构验证错误
        print(ve)


context_str = """
主诉: 咳嗽、咳痰1+月，加重伴发热、呼吸困难8天,1.起病急，病程短。2.1+月前患者受凉后出现咳嗽、咳痰，咳黄色粘稠痰，伴乏力、纳差，无发热、胸痛、咯血，无意识障碍，于药店购买“感冒药”治疗，上述症状缓解。8天前患者咳嗽、咳痰加重，伴呼吸困难、发热、嗜睡、乏力，体温最高38.6°C，就诊于龙泉驿第一人民医院，查血气分析：PH7.29、PCO278mmHg、PO252mmHg，胸部CT：左肺上叶片状模糊影，左肺下叶团块影，大小约5.1×4.5cm，心脏增大、肺动脉增粗。考虑“重症肺炎、急性呼吸窘迫综合征、感染性休克、II型呼吸衰竭、多脏器功能障碍、睡眠呼吸暂停综合征”。给予气管插管、呼吸机辅助通气，哌拉西林他唑巴坦、左氧氟沙星、氟康唑抗感染及对症保肝、护胃等治疗。患者脱机困难、呼吸衰竭难以纠正，于2020.01.20转入我院急诊科。完善辅助检查：血气分析:酸碱度7.246、氧分压68.4mmHg、二氧化碳分压50.1mmHg，血常规:白细胞计数7.30×109/L,中性粒细胞4.79×109/L,血红蛋白171g/L,血小板计数81×109/L，生化：总胆红素36.8umol/L、直接胆红素30.8umol/L、白蛋白25.9g/L、谷氨酰转肽酶71IU/L，CT：双肺散在斑片、条索影，双肺下叶实变不张；心脏增大，肺动脉干稍增粗；纵隔淋巴结增多；脾脏体积增大；胰腺颈体部边缘可疑稍毛糙；脑实质未见明显异常密度灶。给予新特灭抗感染，氨溴索祛痰，补充白蛋白，去甲肾上腺素维持血压。今为进一步诊治收住我科。患者睡眠呼吸暂停伴打鼾7年，每日睡眠约20h，近3年打鼾、睡眠呼吸暂停加重，睡眠时有面色发紫，未诊治。10+年前发现乙肝小三阳，未治疗。3年前于我院诊断“肥胖症”，未服药治疗。9年前因外伤致“右上肢骨折”行“右上肢内固定术”，术后未取出内固定植入物。3.镇痛镇静状态，体型肥胖，全身散在色素沉着伴瘀红,全身散在红疹伴部分破溃,腹股沟及腹部散在潮红糜烂,右上肢可见陈旧手术瘢痕。咽喉部气管插管在位。双下肺叩诊呈浊音，双肺呼吸音粗，双肺均可闻及湿啰音，双侧呼吸运动均匀对称，无增强或者减弱，双肺触觉语颤对称无异常，未触及胸膜摩擦感，胸廓未见异常。双侧乳房对称，未见异常。心界正常，心律齐，各瓣膜区未闻及杂音。腹部饱满，全腹软，腹部未触及包块，肝肋下约2横指，脾肋缘下约1横指，双肾未触及。右上肢外侧可见一长约15cm纵形陈旧性手术疤痕。脑膜刺激征阴性，四肢肌力查体不能配合，双下肢轻度凹陷性水肿，双下肢病理征阳性。4.（2010.01.20）血气分析:酸碱度7.246、氧分压68.4mmHg、二氧化碳分压50.1mmHg，血常规:白细胞计数7.30×109/L,中性粒细胞4.79×109/L,血红蛋白171g/L,血小板计数81×109/L，生化：总胆红素36.8umol/L、直接胆红素30.8umol/L、白蛋白25.9g/L、谷氨酰转肽酶71IU/L、钠134.5mmol/L、钙1.89mmol/L，心肌标志物:尿钠素203ng/L、肌钙蛋白-T32.0ng/L，降钙素原0.14ng/ml，输血前全套:乙肝表面抗原阳性、乙肝e抗体阳性、乙肝核心抗体阳性；CT：双肺散在斑片、条索影，双肺下叶实变不张；心脏增大，肺动脉干稍增粗；纵隔淋巴结增多；脾脏体积增大；胰腺颈体部边缘可疑稍毛糙；脑实质未见明显异常密度灶。（2010.01.21）促肾上腺皮质激素、生长激素、睾酮、性激素结合球蛋白无异常。促甲状腺刺激激素8.900mU/L、催乳素45.91ng/mL、皮质醇（8-10点）139.00nmol/L。真菌G试验、GM试验均为阴性。
"""
score,reason = llm_predict_score(llm,'心脏骤停',context_str)
print(score)
print(reason)


# Symptom_cardiac_arrest_Score: { "score": 0.9,                          "reason":" In this case, the patient is described as having multiple severe health conditions and medical interventions. It mentions "successful resuscitation after cardiac arrest (post-cardiopulmonary resuscitation)," which means the patient experienced one or more events of sudden cardiac stop and regained heartbeat and breathing after being administered cardiopulmonary resuscitation. }

# {
#   "Symptom_cardiac_arrest_Score": {
#     "score": 0.9,
#     "reason": "In this case, the patient is described as having multiple severe health conditions and medical interventions. It mentions 'successful resuscitation after cardiac arrest (post-cardiopulmonary resuscitation),' which means the patient experienced one or more events of sudden cardiac stop and regained heartbeat and breathing after being administered cardiopulmonary resuscitation."
#   }
# }


# {
# """
# Clinically, this plot indicates that the occurrence of cardiac arrest in sepsis patients is a crucial factor 
# for assessing the severity of the condition and the likelihood of adverse outcomes. As the risk score increases 
# (moving up the vertical axis), it becomes more imperative for healthcare professionals to consider aggressive treatment strategies 
# and closer monitoring to mitigate the risks associated with sepsis.
# Additionally, as the confidence in the association increases (moving to the right along the horizontal axis), it further validates cardiac arrest 
# as a significant predictor in sepsis prognosis models. 
# ...
# It is important to note that cardiac arrest is a severe complication in sepsis patients, and its occurrence significantly impacts the patient's prognosis.
# """
# }