# Advanced Data Mining Project

![Data Mining](https://img.shields.io/badge/Data-Mining-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-success)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-important)

A comprehensive data analysis platform implementing four key data mining algorithms with an interactive interface.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithms](#algorithms)
- [Project Structure](#project-structure)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

## Features

- **Four Core Algorithms**:
  - 🛒 Apriori (Association Rule Mining)
  - 🧠 Naive Bayes (Classification)
  - 🌳 ID3 Decision Tree
  - 📊 K-Means Clustering
- Interactive Web UI using Streamlit
- Automatic documentation generation
- Visualizations for all algorithms
- Customizable parameters for each algorithm

## Requirements

- Python 3.8+
- Required Packages:
- streamlit==1.26.0
- pandas==2.0.3
- numpy==1.24.3
- scikit-learn==1.3.0
- mlxtend==0.22.0
- matplotlib==3.7.2
- seaborn==0.12.2

## Installation

1. Clone the repository:

```bash
git clone https://github.com/radanpro/data-mining.git
cd data-mining
```

Create and activate virtual environment (recommended):

```bash
python -m venv venv
```

#### Windows

```bash
venv\Scripts\activate
```

#### Mac/Linux

```bash
source venv/bin/activate
```

## Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

#### Start the application:

```bash
streamlit run app.py
```

## Workflow:

1.  Upload CSV dataset

2.  Navigate through algorithm sections

3.  Adjust parameters using sliders

4.  View interactive results and visualizations

5.  Check generated documentation in documentation.md

#### Example Dataset Format:

Age Income Gender Purchased
25 50000 Male No
30 70000 Female Yes

Algorithms

1. Apriori (Association Rule Mining)
   -Finds relationships between items in transactions
   -Configurable support and confidence levels
   -Outputs rules with lift metric

2. Naive Bayes Classifier
   -Probabilistic classification model
   -Shows accuracy and confusion matrix
   -Displays feature importance

3. ID3 Decision Tree
   -Implements information gain strategy
   -Visualizes decision tree structure
   -Displays classification rules

4. K-Means Clustering
   -Unsupervised clustering algorithm
   -Elbow method visualization
   -Silhouette score evaluation
   -Project Structure

## Project Structure

```
data-mining-project/
├── app.py # Main application code
├── documentation.md # Auto-generated documentation
├── plots/ # Saved visualizations
├── dataset.scv
└── requirements.txt # Dependency list
```

## Contributing

Fork the repository

Create your feature branch (`git checkout -b feature/AmazingFeature`)

Commit your changes (`git commit -m 'Add some AmazingFeature'`)

Push to the branch (`git push origin feature/AmazingFeature`)

Open a Pull Request
<br>
<br>
<br>
<br>

---

<br>
<br>

# README - Data Mining Project

## **1. Dataset Contents**

The dataset contains transactional sales data with the following columns:

1. **Region**: Geographic region where the sale occurred (e.g., Europe, Asia, Sub-Saharan Africa).
2. **Country**: Specific country in the region.
3. **Item Type**: Type of product sold (e.g., Beverages, Baby Food, Vegetables).
4. **Sales Channel**: Sales method used (Online or Offline).
5. **Order Priority**: Priority level of the order (L: Low, M: Medium, C: Critical, H: High).
6. **Order Date**: Date when the order was placed.
7. **Order ID**: Unique identifier for the order.
8. **Ship Date**: Date when the order was shipped.
9. **Units Sold**: Number of units sold.
10. **Unit Price**: Price per unit of the product.
11. **Unit Cost**: Production cost per unit.
12. **Total Revenue**: Total revenue from the sale (Units Sold \* Unit Price).
13. **Total Cost**: Total cost incurred (Units Sold \* Unit Cost).
14. **Total Profit**: Total profit from the sale (Total Revenue - Total Cost).

---

## **2. Project Goal**

The goal of this project is to perform data mining on the provided dataset to:

- **Analyze sales patterns**: Understand the relationship between sales performance and attributes like region, product type, and sales channel.
- **Discover hidden insights**: Use data mining algorithms to uncover meaningful patterns and relationships within the data.
- **Build predictive models**: Develop models to classify and cluster the data for better decision-making.
- **Generate actionable insights**: Provide recommendations to optimize sales strategies and improve profitability.

---

## **3. Expected Results**

1. **Frequent Itemset Analysis**: Using the Apriori algorithm to discover patterns and rules such as "products frequently sold together."
2. **Classification Results**:

- Predict the priority level of orders based on attributes.
- Evaluate model accuracy using metrics like Precision, Recall, and F1-Score.

3. **Cluster Analysis**: Group sales data into clusters for better segmentation (e.g., high-profit vs. low-profit orders).
4. **Insights and Recommendations**: Actionable insights based on analysis results to optimize sales and improve decision-making.

---

## **4. Why Data Mining?**

Data mining is essential for this project to:

- Identify meaningful patterns in complex datasets.
- Provide actionable insights that support data-driven decision-making.
- Improve the efficiency and effectiveness of sales strategies.
- Segment customers or orders for targeted actions.

---

## **5. Task Assignments and Dependencies**

| **Task**                                     | **Description**                                                                                      | **Assigned To** | **Dependencies**                                                              |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------- | --------------- | ----------------------------------------------------------------------------- |
| **Task 1: Data Exploration & Preprocessing** | Load, clean, and preprocess the dataset (handle missing values, normalize data, encode categorical). | Person 1        | Independent task; outputs clean dataset for other tasks.                      |
| **Task 2: Feature Selection & Analysis**     | Analyze data relationships, select relevant features, and provide visualizations.                    | Person 2        | Depends on clean dataset from Person 1.                                       |
| **Task 3: Apriori Algorithm Implementation** | Use the Apriori algorithm to discover association rules and visualize patterns.                      | Person 3        | Requires selected features from Person 2.                                     |
| **Task 4: Naïve Bayes Classification**       | Build a Naïve Bayes model to classify order priorities and evaluate performance metrics.             | Person 4        | Requires selected features from Person 2 and preprocessed data from Person 1. |
| **Task 5: ID3 Algorithm Implementation**     | Develop a decision tree for classification and visualize decision-making logic.                      | Person 5        | Requires selected features from Person 2 and preprocessed data from Person 1. |
| **Task 6: K-Means Clustering**               | Apply K-Means clustering to segment sales data and visualize clusters.                               | Person 6        | Requires selected features from Person 2 and preprocessed data from Person 1. |
| **Task 7: Documentation and Reporting**      | Compile a detailed report covering preprocessing, algorithms, results, and insights.                 | Collaborative   | Depends on contributions and results from all team members.                   |

---

## **6. Notes**

- Collaboration tools such as **Trello** or **Jira** can be used to track progress and ensure smooth communication.
- Code should be documented and modular to ensure reproducibility.
- Final deliverables include a complete report, codebase, and visualizations of results.

<br>
<br>
<br>

---

<br>
<br>
<br>

# README - مشروع تنقيب البيانات

## **1. محتويات البيانات (Dataset Contents)**

تتضمن البيانات معلومات معاملات المبيعات مع الأعمدة التالية:

1. **المنطقة (Region):** المنطقة الجغرافية التي تمت فيها المبيعات (مثل أوروبا، آسيا، إفريقيا جنوب الصحراء).
2. **الدولة (Country):** اسم الدولة المرتبطة بالمبيعات.
3. **نوع المنتج (Item Type):** تصنيف المنتجات (مثل المشروبات، غذاء الأطفال، الخضروات).
4. **قناة البيع (Sales Channel):** طريقة البيع (عبر الإنترنت أو دون اتصال).
5. **أولوية الطلب (Order Priority):** مستوى الأولوية المعطى للطلب (L: منخفض، M: متوسط، C: حرج، H: عالي).
6. **تاريخ الطلب (Order Date):** تاريخ إجراء الطلب.
7. **معرف الطلب (Order ID):** معرف فريد لكل طلب.
8. **تاريخ الشحن (Ship Date):** تاريخ شحن الطلب.
9. **الوحدات المباعة (Units Sold):** عدد الوحدات المباعة.
10. **سعر الوحدة (Unit Price):** سعر الوحدة الواحدة من المنتج.
11. **تكلفة الوحدة (Unit Cost):** تكلفة إنتاج الوحدة.
12. **إجمالي الإيرادات (Total Revenue):** الإيرادات الكلية الناتجة عن المبيعات (الوحدات المباعة × سعر الوحدة).
13. **إجمالي التكلفة (Total Cost):** التكلفة الكلية (الوحدات المباعة × تكلفة الوحدة).
14. **إجمالي الربح (Total Profit):** الفرق بين الإيرادات والتكاليف (إجمالي الإيرادات - إجمالي التكلفة).

---

## **2. الغاية من المشروع (Project Goal)**

الهدف من هذا المشروع هو تنفيذ تنقيب البيانات على البيانات المقدمة من أجل:

- **تحليل أنماط المبيعات:** فهم العلاقة بين أداء المبيعات والخصائص مثل المنطقة، نوع المنتج، وقناة البيع.
- **اكتشاف الأفكار المخفية:** استخدام خوارزميات التنقيب عن البيانات لاستخراج الأنماط والعلاقات المفيدة داخل البيانات.
- **بناء نماذج تنبؤية:** تطوير نماذج لتصنيف وتجميع البيانات لدعم اتخاذ القرارات.
- **تقديم رؤى قابلة للتنفيذ:** تقديم توصيات لتحسين استراتيجيات المبيعات وزيادة الربحية.

---

## **3. النتائج المتوقعة (Expected Results)**

1. **تحليل العناصر المتكررة:** استخدام خوارزمية Apriori لاكتشاف الأنماط والقواعد مثل "المنتجات التي تُباع معًا بشكل متكرر".
2. **نتائج التصنيف:**

- التنبؤ بمستوى أولوية الطلبات بناءً على الخصائص.
- تقييم دقة النموذج باستخدام مقاييس مثل الدقة (Accuracy)، والاستدعاء (Recall)، وF1-Score.

3. **تحليل التجمعات (Clusters):** تقسيم بيانات المبيعات إلى مجموعات لتحسين التقسيم (مثل الطلبات ذات الربح العالي مقابل الربح المنخفض).
4. **رؤى وتوصيات:** تقديم رؤى قابلة للتنفيذ بناءً على نتائج التحليل لتحسين اتخاذ القرارات.

---

## **4. لماذا نحتاج إلى تنقيب البيانات؟ (Why Data Mining?)**

تنقيب البيانات ضروري لهذا المشروع من أجل:

- تحديد الأنماط المفيدة في مجموعات البيانات المعقدة.
- توفير رؤى قابلة للتنفيذ لدعم اتخاذ القرارات المبنية على البيانات.
- تحسين كفاءة وفعالية استراتيجيات المبيعات.
- تقسيم العملاء أو الطلبات لاتخاذ إجراءات مستهدفة.

---

## **5. توزيع المهام والاعتماد بينها (Task Assignments and Dependencies)**

| **المهمة**                                   | **الوصف**                                                                                | **المسؤول**   | **الاعتماد**                                                                           |
| -------------------------------------------- | ---------------------------------------------------------------------------------------- | ------------- | -------------------------------------------------------------------------------------- |
| **المهمة 1: استكشاف ومعالجة البيانات**       | تحميل وتنظيف وإعداد البيانات (التعامل مع القيم المفقودة، تطبيع البيانات، وترميز الفئات). | الشخص الأول   | مهمة مستقلة؛ تُنتج بيانات نظيفة للمهمات الأخرى.                                        |
| **المهمة 2: تحليل البيانات واختيار الميزات** | تحليل العلاقات بين الخصائص، اختيار الميزات المهمة، وتوفير تصورات بيانية.                 | الشخص الثاني  | تعتمد على البيانات النظيفة من الشخص الأول.                                             |
| **المهمة 3: تنفيذ خوارزمية Apriori**         | استخدام خوارزمية Apriori لاكتشاف القواعد الترابطية وتصوير الأنماط.                       | الشخص الثالث  | تعتمد على الميزات المختارة من الشخص الثاني.                                            |
| **المهمة 4: تصنيف Naïve Bayes**              | بناء نموذج Naïve Bayes لتصنيف الأولويات وتقييم الأداء باستخدام مقاييس متعددة.            | الشخص الرابع  | تعتمد على الميزات المختارة من الشخص الثاني والبيانات المُعالجة من الشخص الأول.         |
| **المهمة 5: تنفيذ خوارزمية ID3**             | تطوير شجرة قرارات للتصنيف وتصوير منطق اتخاذ القرار.                                      | الشخص الخامس  | تعتمد على الميزات المختارة من الشخص الثاني والبيانات المُعالجة من الشخص الأول.         |
| **المهمة 6: تنفيذ خوارزمية K-Means**         | تطبيق K-Means لتقسيم البيانات إلى مجموعات وتصوير النتائج.                                | الشخص السادس  | تعتمد على الميزات العددية المختارة من الشخص الثاني والبيانات المُعالجة من الشخص الأول. |
| **المهمة 7: التوثيق وإعداد التقارير**        | إعداد تقرير شامل يغطي مراحل معالجة البيانات، الخوارزميات، النتائج، والرؤى.               | الفريق بأكمله | تعتمد على مساهمات ونتائج جميع أعضاء الفريق.                                            |

---

## **6. ملاحظات (Notes)**

- يمكن استخدام أدوات تعاون مثل **Trello** أو **Jira** لتتبع التقدم وضمان تواصل فعال.
- يجب توثيق الكود وجعله معياريًا لضمان إمكانية إعادة الإنتاج.
- المخرجات النهائية تشمل تقريرًا كاملًا، الشيفرة المصدرية، ورسوم بيانية للنتائج.

```

```
