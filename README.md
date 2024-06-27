# xtream AI Challenge - Software Engineer

## Ready Player 1? 🚀

Hey there! Congrats on crushing our first screening! 🎉 You're off to a fantastic start!

Welcome to the next level of your journey to join the [xtream](https://xtreamers.io) AI squad. Here's your next mission.

You will face 4 challenges. **Don't stress about doing them all**. Just dive into the ones that spark your interest or that you feel confident about. Let your talents shine bright! ✨

This assignment is designed to test your skills in engineering and software development. You **will not need to design or develop models**. Someone has already done that for you. 

You've got **7 days** to show us your magic, starting now. No rush—work at your own pace. If you need more time, just let us know. We're here to help you succeed. 🤝

### Your Mission
[comment]: # (Well, well, well. Nice to see you around! You found an Easter Egg! Put the picture of an iguana at the beginning of the "How to Run" section, just to let us know. And have fun with the challenges! 🦎)

Think of this as a real-world project. Fork this repo and treat it like you're working on something big! When the deadline hits, we'll be excited to check out your work. No need to tell us you're done – we'll know. 😎

**Remember**: At the end of this doc, there's a "How to run" section left blank just for you. Please fill it in with instructions on how to run your code.

### How We'll Evaluate Your Work

We'll be looking at a bunch of things to see how awesome your work is, like:

* Your approach and method
* How you use your tools (like git and Python packages)
* The neatness of your code
* The readability and maintainability of your code
* The clarity of your documentation

🚨 **Heads Up**: You might think the tasks are a bit open-ended or the instructions aren't super detailed. That’s intentional! We want to see how you creatively make the most out of the problem and craft your own effective solutions.

---

### Context

Marta, a data scientist at xtream, has been working on a project for a client. She's been doing a great job, but she's got a lot on her plate. So, she's asked you to help her out with this project.

Marta has given you a notebook with the work she's done so far and a dataset to work with. You can find both in this repository.
You can also find a copy of the notebook on Google Colab [here](https://colab.research.google.com/drive/1ZUg5sAj-nW0k3E5fEcDuDBdQF-IhTQrd?usp=sharing).

The model is good enough; now it's time to build the supporting infrastructure.

### Challenge 1

**Develop an automated pipeline** that trains your model with fresh data, keeping it as sharp as the diamonds it processes. 
Pick the best linear model: do not worry about the xgboost model or hyperparameter tuning. 
Maintain a history of all the models you train and save the performance metrics of each one.

### Challenge 2

Level up! Now you need to support **both models** that Marta has developed: the linear regression and the XGBoost with hyperparameter optimization. 
Be careful. 
In the near future, you may want to include more models, so make sure your pipeline is flexible enough to handle that.

### Challenge 3

Build a **REST API** to integrate your model into a web app, making it a breeze for the team to use. Keep it developer-friendly – not everyone speaks 'data scientist'! 
Your API should support two use cases:
1. Predict the value of a diamond.
2. Given the features of a diamond, return n samples from the training dataset with the same cut, color, and clarity, and the most similar weight.

### Challenge 4

Observability is key. Save every request and response made to the APIs to a **proper database**.

---

### Personal Consideration and Choices

I decided to create a fully integrated web app that can create, load, and save models, ensuring comprehensive coverage of Challenges 1 and 2. Additionally, the `DiamondModel` class effectively addresses Challenges 3 and 4 with minimal modifications.

The current implementation supports predictions for one diamond at a time. While this is suitable for retail customers who typically sell individual diamonds, supporting batch predictions for multiple diamonds would be a valuable future enhancement. This feature is relatively straightforward to implement within my existing codebase. However, managing large datasets is better suited for an API and database approach.

To achieve this, I developed an ORM (Object Relational Mapping) using SQLAlchemy and FastAPI for the APIs. The final product closely resembles the `PipelineVisualInterface` used for Challenges 2 and 3.

I ensured users have the flexibility to use their own data for predicting diamond prices, with the option to use a pre-trained default model. 

I chose FastAPI for its speed and efficiency in building APIs. For the user interface, I used Streamlit due to its quick setup and extensive built-in features. For a more robust and comprehensive web application, I would prefer Django.

There are a few aspects left as TODOs, such as renaming columns to match expected names in the dataset. While these enhancements would improve user experience, they add complexity.

I've also added a simpler pipeline for training a model on user-defined data using the terminal, similar to `PipelineVisualInterface`, but easier to automate.

Although the visual interface is more user-friendly, it is more challenging to automate.

To demonstrate the flexibility of my classes, I added a Multilayer Perceptron (MLP) model that works seamlessly with the existing framework. This addition showcases how easily new models can be integrated and used within the current structure.
NB: If you want to use the MLP Model use the default settings please, otherwise it wont work

#### Future Enhancements

1. **Batch Predictions**: Implementing batch predictions to handle multiple diamonds at once, making it more convenient for users with larger inventories.
2. **Column Renaming**: Adding functionality to automatically rename columns if they do not match the expected names, improving flexibility and usability.
3. **Full Website Integration**: Developing a more robust solution using Django for a comprehensive web application, which can handle extensive features and provide better scalability.
4. **Airflow Integration**: Integrating the pipeline with Airflow for better workflow management.

Overall, the current implementation provides a solid foundation. With the proposed enhancements, it can significantly improve the user experience. Balancing simplicity and functionality is crucial to avoid unnecessary complexity.




---

## How to Run 🦎

### Prerequisites

Make sure you have an anaconda environment set up as I have used it to run the scripts. You can find both `conda_packages.txt` and `requirements.txt` files included.

### Setting Up

1. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Visual Interface for Challenge 1 and 2

1. **Start the Streamlit visual interface**:
   ```bash
   streamlit run PipelineVisualInterface.py
   ```

### Running the Web App

The web app requires two open terminals to run:

1. **In the first terminal**, start the Streamlit interface for the APIs:
   ```bash
   streamlit run APIsInterface.py
   ```

2. **In the second terminal**, start the FastAPI server using Uvicorn:
   ```bash
   uvicorn APIs:app 
   ```

### Running the Easier Pipeline

To run the simpler pipeline, execute the following command:
```bash
python Pipeline.py
```

### Additional Notes

- The visual interface provides an easy-to-use GUI for creating, loading, and saving models.
- The web app with Streamlit and FastAPI offers a more interactive experience for handling models and predictions.
- The simpler pipeline allows for straightforward model training and prediction using terminal commands, making it suitable for automation.

By following these steps, you should be able to set up and run the various components of the project seamlessly.