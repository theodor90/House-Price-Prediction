# House Price Prediction

This project uses ML.NET to build a regression model that predicts house sale prices based on features such as house size, number of bedrooms, number of bathrooms, and neighborhood.

## Project Overview

The solution demonstrates the following steps:

1. **Data Loading:**  
   Reads a CSV file (`house-price-data.csv`) into an `IDataView`, where each record is mapped to the `HouseData` class.

2. **Data Splitting:**  
   Splits the dataset into a training set (80%) and a testing set (20%) for model evaluation.

3. **Pipeline Creation:**  
   - **Feature Engineering:**  
     Concatenates numeric features and applies one-hot encoding to the categorical `Neighborhood` feature.
   - **Label Assignment:**  
     Uses the `SalePrice` column as the label.
   - **Model Training:**  
     Fits a FastTree regression model.

4. **Model Evaluation:**  
   (Commented out in the code) Evaluates the model using regression metrics such as R² and Root Mean Squared Error.

5. **Prediction:**  
   Demonstrates making a prediction using a prediction engine for a given sample house data.

## Prerequisites

- [.NET 8 SDK](https://dotnet.microsoft.com/en-us/download/dotnet/8.0)
- [Visual Studio 2022](https://visualstudio.microsoft.com/)
- [ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet) (Installed via NuGet package: `Microsoft.ML`)

## How to Run

1. **Clone the Repository:**  
   Clone this repository to your local machine.

2. **Prepare the Dataset:**  
   Ensure that the `house-price-data.csv` file is placed in the same directory as the executable (or modify the `dataPath` in the code as needed).

3. **Open the Project in Visual Studio:**  
   Open the solution using Visual Studio 2022.

4. **Restore NuGet Packages:**  
   Restore the required packages via __Solution Explorer > Right-Click on the project > Manage NuGet Packages__.

5. **Build and Run:**  
   - Build the solution.
   - Run the project.  
   The application will output the predicted sale price to the console.

## Code Characteristics

- **Language:** C# 12.0
- **Target Framework:** .NET 8

## Notes

- The evaluation metric output lines are commented out. Uncomment these lines if you wish to see the R² score and the Root Mean Squared Error.
- Adjust the data file path in `Program.cs` if your CSV file is located elsewhere.

## Acknowledgements

This project leverages the capabilities of [ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet) for machine learning and demonstrates a simple yet effective approach for regression-based predictions.


