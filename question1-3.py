# Question 1: 

# a) College Completion Dataset
# Do private colleges have higher student completion rates than public colleges?
# b) Job Placement Dataset
# Are students with prior work experience more financially successful in the workforce than those without?

# Question 2: 

# a) College Completion Dataset
# - Question: Do private colleges have higher student completion rates than public colleges?
# - Independent business metric: Awards per 100 full-time undergraduate students
# Data preparation
# - correct variable type/class as needed
#   - control: categorical (public/private)
#   - awards_per_value: numeric (float)
# - collapse factor levels as needed
#   - 'Private not-for-profit', 'Private for-profit' --> 'Private'
# - one-hot encoding factor variables
#   - Public, Private --> 0, 1
# - normalize the continuous variables
#   - awards_per_value
# - drop unneeded variables
#   - everything except control & awards_per_value
# - create target variable if needed
#   - awards_per_value already represents target variable (completion rate)
# - calculate the prevalence of the target variable
# - create the necessary data partitions (Train,Tune,Test)

# b) Job Placement Dataset
# - Question: Are students with prior work experience more financially successful in the workforce than those without?
# - Independent business metric: Salary
# Data preparation
# - correct variable type/class as needed
#   - workex: categorical (yes/no)
#   - salary: numeric (float/integer)
# - collapse factor levels as needed
#   - N/A: workex only has two levels
# - one-hot encoding factor variables
#   - Yes, No --> 1, 0
# - normalize the continuous variables
#   - salary
# - drop unneeded variables
#   - everything except workex & salary
# - create target variable if needed
#   - salary already represents target variable (financial success)
# - calculate the prevalence of the target variable
# - create the necessary data partitions (Train,Tune,Test)

# Question 3:

# a) College Completion Dataset
# The dataset seems to be suitable to answer my question, as it contains the minimum necessary variables- control (public/private), and a numeric proxy for student success (awards_per_value). My first instinct is that private colleges may have higher completion rates due to their smaller size and increased resources, and there are also several areas of caution to note. This includes potential outliers, impact of cohort sizes, missing data, differences in program types, and lack of context. 
# b) Job Placement Dataset
# The dataset contains the necessary variables to answer my question- workex and salary. My initial expectation is that students with prior work experience do, on average, earn higher salaries than students without. However, potential concerns include missing data, skewed distribution, lack of context regarding prior work experience, and other confounding factors such as academic performance or degree type. 
