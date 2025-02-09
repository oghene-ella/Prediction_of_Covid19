# Prediction of Economic Impact of COVID-19 in Sub-Saharan Africa

## Abstract

The global economy has faced an unprecedented threat from the COVID-19 outbreak. The current illness is ravaging Africa at an exponential rate, and all of the continent's nations are experiencing the effects of the epidemic. In this study, we used the COVID-19 outbreak as a case study to examine the prediction of the economic impact of the pandemic in Sub-Saharan Africa (SSA). We adopted several models to work on the dataset to compare which one worked best for all three target variables and discovered that Gradient Boosting performed the best across all models. We used the gradient boosting approach to explain the interaction between economic agents and how the COVID-19 outbreak has affected their relationships. We examined the effects of COVID-19 on economic variables and found that Brand Purchase, Food Amount, and Food Shopping were statistically significant at 5% and 10%, respectively.

## Table of Contents

- [Project Overview](#project-overview)
- [Models and Methodology](#models-and-methodology)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

## Project Overview

This project aims to predict the economic impact of the COVID-19 pandemic on Sub-Saharan Africa (SSA). Using various machine learning models, we explored and compared their performance on predicting the effects of COVID-19 on economic variables like brand purchase behavior, food amount purchased, and food shopping habits. By analyzing the interaction between economic agents during the pandemic, we gained insights into the pandemic's effects on SSA's economy.

## Models and Methodology

In this project, we implemented and compared the performance of the following machine learning models:

- **Linear Regression**
- **Random Forest**
- **Gradient Boosting**

We evaluated these models using cross-validation and determined that **Gradient Boosting** outperformed the others across all three target variables. The key steps involved in the methodology include:

1. **Data Preprocessing**: Cleaning and transforming the dataset to handle missing values and normalize the data.
2. **Model Training**: Training the models using the preprocessed data.
3. **Evaluation**: Comparing the performance of models based on metrics like R-squared, Mean Squared Error (MSE), and significance of the target variables.

## Results

- **Best Model**: Gradient Boosting
- **Significant Variables**: 
  - Brand Purchase (statistically significant at 5%)
  - Food Amount (statistically significant at 5%)
  - Food Shopping (statistically significant at 10%)

The Gradient Boosting model showed the best performance in predicting economic behaviors, providing key insights into the impact of COVID-19 on SSA's economy.

## Conclusion

This project sheds light on the economic disruptions caused by COVID-19 in Sub-Saharan Africa. By leveraging machine learning techniques, we successfully predicted the impacts on various economic behaviors. The findings have practical applications for policymakers and businesses in understanding and mitigating the effects of future pandemics or crises.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
