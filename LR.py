import pandas as pd
from sklearn.linear_model import LogisticRegression
import streamlit as st

# 读取训练集数据
train_data = pd.read_csv('train_data.csv')

# 分离输入特征和目标变量
X = train_data[['Age', 'Sex', 'Race', 'Marital status',
                'T stage', 'N stage', 'Surgery', 'Radiation',
                'Chemotherapy']]
y = train_data['Vital status']

# 特征映射
class_mapping = {0: "Alive", 1: "Dead"}
age_mapper = {'≤65': 1, '>65': 2}
sex_mapper = {'female': 1, 'male': 2}
race_site_mapper = {"White": 1, "Black": 2, "Other": 3}
marital_status_mapper = {"Married": 1, "Other": 2}
t_stage_mapper = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
n_stage_mapper = {"N0": 1, "N2": 2, "N3": 3, "N4": 4}
surgery_mapper = {"NO": 0, "Yes": 1}
radiation_mapper = {"NO": 0, "Yes": 1}
chemotherapy_mapper = {"NO": 0, "Yes": 1}

# 将分类变量映射为数值
X['Age'] = X['Age'].map(age_mapper)
X['Sex'] = X['Sex'].map(sex_mapper)
X['Race'] = X['Race'].map(race_site_mapper)
X['Marital status'] = X['Marital status'].map(marital_status_mapper)
X['T stage'] = X['T stage'].map(t_stage_mapper)
X['N stage'] = X['N stage'].map(n_stage_mapper)
X['Surgery'] = X['Surgery'].map(surgery_mapper)
X['Radiation'] = X['Radiation'].map(radiation_mapper)
X['Chemotherapy'] = X['Chemotherapy'].map(chemotherapy_mapper)

# 创建并训练逻辑回归模型
lr_model = LogisticRegression()
lr_model.fit(X, y)

# 预测函数
def predict_Vital_status(age, sex, race, marital_status, t_stage, n_stage, surgery, radiation, chemotherapy):
    input_data = pd.DataFrame({
        'Age': [age_mapper[age]],
        'Sex': [sex_mapper[sex]],
        'Race': [race_site_mapper[race]],
        'Marital status': [marital_status_mapper[marital_status]],
        'T stage': [t_stage_mapper[t_stage]],
        'N stage': [n_stage_mapper[n_stage]],
        'Surgery': [surgery_mapper[surgery]],
        'Radiation': [radiation_mapper[radiation]],
        'Chemotherapy': [chemotherapy_mapper[chemotherapy]],
    })
    prediction = lr_model.predict(input_data)[0]
    probability = lr_model.predict_proba(input_data)[0][1]  # 获取属于类别1的概率
    class_label = class_mapping[prediction]
    return class_label, probability

# 创建Web应用程序
st.title("6-month survival of stage III sclc patients based on Logistic Regression")
st.sidebar.write("Variables")

age = st.sidebar.selectbox("Age", options=list(age_mapper.keys()))  # 使用选择框
sex = st.sidebar.selectbox("Sex", options=list(sex_mapper.keys()))
race = st.sidebar.selectbox("Race", options=list(race_site_mapper.keys()))
marital_status = st.sidebar.selectbox("Marital status", options=list(marital_status_mapper.keys()))
t_stage = st.sidebar.selectbox("T stage", options=list(t_stage_mapper.keys()))
n_stage = st.sidebar.selectbox("N stage", options=list(n_stage_mapper.keys()))
surgery = st.sidebar.selectbox("Surgery", options=list(surgery_mapper.keys()))
radiation = st.sidebar.selectbox("Radiation", options=list(radiation_mapper.keys()))
chemotherapy = st.sidebar.selectbox("Chemotherapy", options=list(chemotherapy_mapper.keys()))

if st.button("Predict"):
    prediction, probability = predict_Vital_status(
        age, sex, race, marital_status, t_stage, n_stage, surgery, radiation, chemotherapy
    )

    st.write("Predicted Vital Status:", prediction)
    st.write("Probability of 6-month survival is:", probability)
