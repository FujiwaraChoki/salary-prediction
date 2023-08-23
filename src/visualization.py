import matplotlib.pyplot as plt
import seaborn as sns


def visualize_gender_distribution(data):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Gender', data=data)
    plt.title('Distribution of Genders')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.show()


def visualize_department_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Department', data=data,
                  order=data['Department'].value_counts().index)
    plt.title('Distribution of Departments')
    plt.xlabel('Department')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# Define other visualization functions here...
