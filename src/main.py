import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data_processing import load_data
from visualization import visualize_gender_distribution, visualize_department_distribution

# Define the neural network model for salary prediction


class SalaryPredictionModel(nn.Module):
    def __init__(self):
        super(SalaryPredictionModel, self).__init__()
        # One input feature (age) and one output (salary)
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def preprocess_data(data):
    age_tensor = torch.tensor(
        data['Age'].values, dtype=torch.float32).view(-1, 1)
    salary_tensor = torch.tensor(
        data['Annual Salary'].values, dtype=torch.float32).view(-1, 1)

    age_mean = age_tensor.mean()
    age_std = age_tensor.std()
    salary_mean = salary_tensor.mean()
    salary_std = salary_tensor.std()

    age_tensor = (age_tensor - age_mean) / age_std
    salary_tensor = (salary_tensor - salary_mean) / salary_std

    return age_tensor, salary_tensor, age_mean, age_std, salary_mean, salary_std


def train_model(model, age_tensor, salary_tensor, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(age_tensor)
        loss = criterion(outputs, salary_tensor)
        loss.backward()
        optimizer.step()


def predict_future_salaries(model, future_ages, age_mean, age_std, salary_mean, salary_std):
    future_ages_normalized = (future_ages - age_mean) / age_std
    model.eval()
    future_salaries_normalized = model(future_ages_normalized)
    future_salaries = future_salaries_normalized * salary_std + salary_mean
    return future_salaries


def visualize_predicted_salaries(future_ages, future_salaries):
    plt.figure(figsize=(8, 6))
    plt.plot(future_ages.numpy(), future_salaries.detach().numpy(),
             marker='o', linestyle='-', color='b')
    plt.title('Predicted Future Salaries')
    plt.xlabel('Age')
    plt.ylabel('Predicted Salary')
    plt.grid(True)
    plt.show()


def main():
    data = load_data('employee_data.csv')

    visualize_gender_distribution(data)
    visualize_department_distribution(data)
    # Call other visualization functions here...

    age_tensor, salary_tensor, age_mean, age_std, salary_mean, salary_std = preprocess_data(
        data)

    model = SalaryPredictionModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    num_epochs = 1000
    train_model(model, age_tensor, salary_tensor,
                criterion, optimizer, num_epochs)

    future_ages = torch.tensor([30, 40, 50], dtype=torch.float32).view(-1, 1)
    future_salaries = predict_future_salaries(
        model, future_ages, age_mean, age_std, salary_mean, salary_std)

    print("\nPredicted Future Salaries:")
    for i in range(len(future_ages)):
        print(
            f"For age {future_ages[i].item()}: Predicted salary ${future_salaries[i].item():.2f}")

    visualize_predicted_salaries(future_ages, future_salaries)


if __name__ == "__main__":
    main()
